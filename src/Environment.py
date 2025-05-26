import os
import time
from copy import deepcopy
import random
import torch

from env.scene import list_cooking_appliances, list_chopping_surfaces, list_of_dining_locations, key_from_scene, verify_scene
from sentence_transformers import SentenceTransformer
from sentence_transformers import util as st_utils
from src.utils import *

global embedding_cache_
embedding_cache_ = None
try:
    embedding_cache_ = torch.load("models/_sentence_embeddings_minilm.pt", map_location=DEVICE, weights_only=False)
except:
    print("Trying to read models/_sentence_embeddings_minilm.pt again...")

MATCH_USING_SIMILARITY = True
if MATCH_USING_SIMILARITY:
    similarity_thresh = 0.4
    sentence_model = SentenceTransformer("sentence-transformers/paraphrase-MiniLM-L3-v2", device=DEVICE)
    embed = lambda x: sentence_model.encode(x, batch_size=1, convert_to_tensor=True).detach()

RELAXED_COOK_CHECK=True

def obj_desc(obj_name, obj_dict, no_ids=False):
    description = ""
    if not no_ids:
        description = f"{obj_name} ("
    if obj_dict['description'] != '':
        description += obj_dict['description']
    if len(obj_dict['state']) > 0:
        description += f", {', '.join(obj_dict['state'])}"
    if not no_ids:
        description += ')'
    return description

def parse_action(action_string):
    action_string = action_string.lower()
    num_actions_matched = 0
    ## check which action option matches the action string
    def check_match(action_option, action_string):
        action_specific_words = action_option.lower().split('<x>')
        arguments = []
        for word_idx, action_specific_word in enumerate(action_specific_words):
            if action_specific_word not in action_string:
                return None
            if word_idx == len(action_specific_words)-1:
                arguments.append(action_string.strip())
                break
            arguments.append(action_string[:action_string.find(action_specific_word)].strip())
            action_string = action_string[action_string.find(action_specific_word) + len(action_specific_word):].strip()
        return arguments
    for action_name, action_template in action_options.items():
        arguments = check_match(action_template.lower().strip(), deepcopy(action_string).lower().strip())
        if arguments is not None:
            assert len(arguments[1:]) == action_template.lower().count('<x>'), f"Action {action_name} requires {action_template.lower().count('<x>')} arguments, but got {len(arguments[1:])}"
            num_actions_matched += 1
            yield action_name, arguments[1:]
    ## hacky insert for freeform without 'the object'
    if 'to get' in action_string and num_actions_matched==0:
        action_string_copy = deepcopy(action_string)
        arguments = ["","",""]
        arguments[2] = action_string_copy.split('to get')[1].strip()
        if ' ' in arguments[2]: arguments[2] = arguments[2][:arguments[2].index(' ')]
        action_string_copy = action_string_copy.split('to get')[0].strip()
        if ' ' not in action_string_copy: return
        arguments[0] = action_string_copy.split()[0]
        arguments[1] = ' '.join(action_string_copy.split()[1:])
        if 'the object' in arguments[1]:
            arguments[1].replace('the object','').replace('  ','')
            yield 'freeform', arguments
        elif 'items in' in arguments[1]:
            arguments[1].replace('items in','').replace('  ','')
            yield 'freeform_contents', arguments
        else:
            yield 'freeform', arguments
    return


class Environment():
    def __init__(self, initial_scene):
        self.full_scene = deepcopy(initial_scene)
        self.prev_state = None
        verify_scene(self.full_scene)
        self.action_buffer = []
        self.actions_performed = []
        self.entities_created = {}
        self.objects_used = []
        self.objects_cooked = []
        self.serving_order = []
        self.mixtures = []
        self.transformations = {k:[] for k in self.full_scene.keys()}
        self.step_num = 0
        self.invalid_actions = {}
        if MATCH_USING_SIMILARITY:
            global embedding_cache_
            object_embeddings_dict = embedding_cache_
            if object_embeddings_dict is None:
                embedding_cache_file = "models/_sentence_embeddings_minilm.pt"
                print(f"Reading embeddings from {embedding_cache_file}")
                if os.path.exists(embedding_cache_file):
                    for retries in range(10):
                        try:
                            object_embeddings_dict = torch.load(embedding_cache_file, map_location=DEVICE, weights_only=False)
                            break
                        except:
                            print(f"Trying to read {embedding_cache_file} again...")
                            time.sleep(random.random())
                    embedding_cache_ = object_embeddings_dict
                else:
                    object_embeddings_dict = dict()
            
            for obj_desc in self.full_scene.values():
                if obj_desc['description'] not in object_embeddings_dict:
                    print(f"Embedding {obj_desc['description']}")
                    object_embeddings_dict[obj_desc['description']] = embed([obj_desc['description']])
            self.object_embeddings = torch.cat([object_embeddings_dict[obj_desc['description']] for obj_desc in self.full_scene.values()], dim=0)

    def match_object_through_similarity(self, object_name):
        def match_object(object_name, object_name_in_scene):
            if object_name in object_name_in_scene:
                return True
            elif object_name_in_scene in object_name:
                return True
            else:
                return False
        matches = []
        for object_name_in_scene, object_details in self.full_scene.items():
            if match_object(object_name_in_scene, object_name):
                matches.append(object_name_in_scene)
        random.shuffle(matches)
        matches_secondary = []
        if MATCH_USING_SIMILARITY and len(matches) == 0:
            if object_name == 'granola':
                object_name = 'cereal'
            object_embedding = embed([object_name])
            similarities = st_utils.cos_sim(object_embedding, self.object_embeddings)
            matches_secondary += [obj for obj,sim in zip(self.full_scene.keys(), similarities.view(-1)) if sim > similarity_thresh]
        matches_secondary.sort(key = lambda x: x[1], reverse=True)
        return matches, matches_secondary

    def _check_existence(self, entity):
        entity = entity.strip()
        if entity in self.full_scene.keys():
            return True, ""
        elif ' ' in entity:
            for an_entity in entity.split(' '):
                if an_entity in self.full_scene.keys():
                    return False, f"Action can be performed on only one entity, e.g. {an_entity} at a time. Did you mean <X> {an_entity} <X>"
        matches_primary, matches_secondary = self.match_object_through_similarity(entity)
        matches = matches_primary + matches_secondary
        full_matches = [f"{obj_desc(obj, desc)}" for obj,desc in self.full_scene.items() if obj in matches]
        message = f"Action Failed! {entity} not found."
        if len(full_matches) > 0:
            message += f" Did you mean {' or '.join(full_matches)}? Remember to move or pour contents before using them."
        return False, message

    def _compose_example(self, action, arguments):
        action_template = action_options[action].split('<X>')
        example_action = [f" {a.strip()} {t.strip()}" for a,t in zip(['']+arguments, action_template)]
        return ' '.join(example_action)

    def prompt_string(self, no_ids=False, furniture_only=False, concise=False):
        if concise and not furniture_only:
            raise ValueError("Concise mode is only available for furniture_only=True")
        if concise:
            layout = ''
            rooms = [k for k,v in self.full_scene.items() if v['location'] == 'home']
            for room in rooms:
                furniture = ', '.join([k for k,v in self.full_scene.items() if v['location'] == room])
                layout += f"{room} containing {furniture}; \n"
            return layout[:-3]+'.'
        list_of_strings = []
        def print_children(parent, depth):
            if furniture_only and depth > 1: return
            children = [k for k,v in self.full_scene.items() if v['location'] == parent]
            for child in children:
                list_of_strings.append(f"{'  ' * depth}- {obj_desc(child, self.full_scene[child], no_ids=no_ids)}")
                print_children(child, depth+1)
        print_children('home', 0)
        return '\n'.join(list_of_strings)

    def get_key(self):
        return key_from_scene(self.full_scene)

    def _get_contents(self, container_name, edible_only=False):
        state_contents = ([s.replace('contains','').strip() for s in self.full_scene[container_name]['state'] if 'contains' in s])
        if edible_only:
            entity_contents = ([obj for obj,desc in self.full_scene.items() if desc['location'] == container_name and ('edible' in self.full_scene[obj]['type'] or 'content' in desc['type'])])
        else:
            entity_contents = ([obj for obj,desc in self.full_scene.items() if desc['location'] == container_name])
        return ', '.join(state_contents + entity_contents)


    def _remove_contents(self, container_name):
        ## TODO: What if there's say a spoon in the container
        self.full_scene[container_name]['state'] = [s for s in self.full_scene[container_name]['state'] if 'contains' not in s]
        objects = [obj for obj,desc in self.full_scene.items() if desc['location'] == container_name and ('edible' in desc['type'] or 'content' in desc['type'])]
        for obj in objects:
            del self.full_scene[obj]

    def _deduplicate_name(self, obj_name):
        if obj_name not in self.full_scene.keys():
            try:
                int(obj_name[-1])
                return obj_name
            except:
                pass
        for idx in range(100):
            if obj_name+f'_{idx}' not in self.full_scene.keys():
                return obj_name+f'_{idx}'

    def open(self, arguments):
        object_name = arguments[0]
        found, msg = self._check_existence(object_name)
        if not found: 
            if '<X>' in msg:
                entity = msg.split('<X>')[1].strip()
                arguments[0] = entity
                msg = msg.split('<X>')[0] + self._compose_example('open', arguments)
            return (False, msg)
        if "closed" in self.full_scene[object_name]['state']: self.full_scene[object_name]['state'].remove("closed")
        self.full_scene[object_name]['state'].append("open")
        return (True, f"{object_name} is now open")

    def close(self, arguments):
        object_name = arguments[0]
        found, msg = self._check_existence(object_name)
        if not found: 
            if '<X>' in msg:
                entity = msg.split('<X>')[1].strip()
                arguments[0] = entity
                msg = msg.split('<X>')[0] + self._compose_example('close', arguments)
            return (False, msg)
        if "open" in self.full_scene[object_name]['state']: self.full_scene[object_name]['state'].remove("open")
        self.full_scene[object_name]['state'].append("closed")
        return (True, f"{object_name} is now closed")

    def heat(self, arguments):
        object_name = arguments[0]
        found, msg = self._check_existence(object_name)
        if not found: 
            if '<X>' in msg:
                entity = msg.split('<X>')[1].strip()
                arguments[0] = entity
                msg = msg.split('<X>')[0] + self._compose_example('close', arguments)
            return (False, msg)
        if self.full_scene[object_name]['location'] not in list_cooking_appliances and object_name not in list_cooking_appliances:
            return (False, f"Action failed! Cannot heat {object_name}, because it is not on a cooking appliance.")
        self.full_scene[object_name]['state'].append("hot")
        if object_name not in self.objects_used:
            self.objects_used.append(object_name)
        self._set_cooked(object_name)
        return (True, f"{object_name} is now hot")

    def _set_cooked(self, object_name):
        contents = self._get_contents(object_name, edible_only=True)
        self.objects_cooked.append(object_name)
        self.objects_cooked += contents.split(', ')

    def turn_on(self, arguments):
        object_name = arguments[0]
        found, msg = self._check_existence(object_name)
        if not found: 
            if '<X>' in msg:
                entity = msg.split('<X>')[1].strip()
                arguments[0] = entity
                msg = msg.split('<X>')[0] + self._compose_example('close', arguments)
            return (False, msg)
        if object_name not in list_cooking_appliances:
            return (False, f"Cannot turn on {object_name}, because it is not a cooking appliance.")
        self.full_scene[object_name]['state'].append("on")
        if object_name not in self.objects_used:
            self.objects_used.append(object_name)
        return (True, f"{object_name} is now on")
    
    def turn_off(self, arguments):
        object_name = arguments[0]
        found, msg = self._check_existence(object_name)
        if not found: 
            if '<X>' in msg:
                entity = msg.split('<X>')[1].strip()
                arguments[0] = entity
                msg = msg.split('<X>')[0] + self._compose_example('close', arguments)
            return (False, msg)
        if object_name not in list_cooking_appliances:
            return (False, f"Cannot turn off {object_name}, because it is not a cooking appliance.")
        self.full_scene[object_name]['state'].append("off")
        if object_name not in self.objects_used:
            self.objects_used.append(object_name)
        return (True, f"{object_name} is now off")

    def search(self, arguments):
        container_name = arguments[0]
        found, msg = self._check_existence(container_name)
        if not found: 
            if '<X>' in msg:
                entity = msg.split('<X>')[1].strip()
                arguments[0] = entity
                msg = msg.split('<X>')[0] + self._compose_example('search', arguments)
            return (False, msg)
        matches = [f"{obj_desc(obj, self.full_scene[obj])}" for obj in self.full_scene if self.full_scene[obj]['location'] == container_name]
        return (True, f"Found {', '.join(matches)}")

    def search_to_find(self, arguments):
        container_name = arguments[0]
        found, msg = self._check_existence(container_name)
        if not found: 
            if '<X>' in msg:
                entity = msg.split('<X>')[1].strip()
                arguments[0] = entity
                msg = msg.split('<X>')[0] + self._compose_example('search_to_find', arguments)
            return (False, msg)
        matched_objects = [obj for obj in self.full_scene if self.full_scene[obj]['location'] == container_name]
        matches = [f"{obj_desc(obj, self.full_scene[obj])}" for obj in matched_objects]
        matches += [f"{obj_desc(obj, self.full_scene[obj])}" for obj in self.full_scene if self.full_scene[obj]['location'] == container_name and obj not in matched_objects]
        return (True, f"Found {', '.join(matches)}")

    def look_for(self, arguments):
        object_name = arguments[0]
        if object_name.endswith('s'): object_name = object_name[:-1]
        # Check if the object exists
        objects_matched, objects_matched_secondary = self.match_object_through_similarity(object_name)
        if len(objects_matched) == 0:
            if len(objects_matched_secondary) == 0:
                return (False, f"Action Failed! Object {object_name} not found. You can try using 'search' to find it in a specific location, or use a different word to 'look for' the object, but remember that it can be a single word only.")
            else:
                matches = [f"{obj_desc(obj, self.full_scene[obj])} in/on {self.full_scene[obj]['location']}" for obj in objects_matched_secondary]
                return (True, f"{object_name} not found, but found {', '.join(matches)}. If you don't see what you want, you can try using 'search' to find it in a specific location, or use a different word to 'look_for' the object.")
        else:
            matches = [f"{obj_desc(obj, self.full_scene[obj])} in/on {self.full_scene[obj]['location']}" for obj in objects_matched]
            return (True, f"Found {', '.join(matches)}")

    def find(self, arguments):
        object_name = arguments[0]
        return self.look_for([object_name])

    def move(self, arguments):
        object_name, to_location = arguments
        found, msg = self._check_existence(to_location)
        if not found: 
            if '<X>' in msg:
                entity = msg.split('<X>')[1].strip()
                arguments[1] = entity
                msg = msg.split('<X>')[0] + self._compose_example('move', arguments)
            return (False, msg)
        found, msg = self._check_existence(object_name)
        if not found: 
            if '<X>' in msg:
                entity = msg.split('<X>')[1].strip()
                arguments[0] = entity
                msg = msg.split('<X>')[0] + self._compose_example('move', arguments)
            return (False, msg)
        if any(["contains" in s for s in self.full_scene[object_name]['state']]):
            content = self._get_contents(object_name, edible_only=True).split(', ')
            if len(content) > 1:
                print(f"WARNING in overriding move. More than one contents found in {object_name} state({self.full_scene[object_name]['state']}): {content}. Using the first one.")
            # TODO Maithili: Move all contents maybe?
            content_name = content[0]
            # if VERBOSE:
            #     print('Overriding moving food container to moving content...')
            return self.move_from([content_name, object_name, to_location])
        # Check if the object is not in the to_location
        if to_location == self.full_scene[object_name]['location']: return (True, f"No changes made, {object_name} already at {to_location}")
        
        # Switch the object's location
        current_location = self.full_scene[object_name]['location']
        moving_something_edible =('edible' in self.full_scene[object_name]['type'] or 'content' in self.full_scene[object_name]['type']) and 'container' in self.full_scene[to_location]['type']
        moving_to_someplace_edible = 'edible' in self.full_scene[to_location]['type'] or 'content' in self.full_scene[to_location]['type']
        if moving_to_someplace_edible:
            self.mixtures.append({'container':to_location, 'contents':[to_location]})
        if moving_something_edible or moving_to_someplace_edible:
            added_in_existing_mixture = False
            idx_mix_pop = None
            for idx_mix, mixture in enumerate(self.mixtures):
                if mixture['container'] == to_location:
                    if object_name not in self.mixtures[idx_mix]['contents']:
                        self.mixtures[idx_mix]['contents'].append(object_name)
                    for content in self.mixtures[idx_mix]['contents']:
                        if content in self.entities_created:
                            self.entities_created[content] = self.step_num
                    added_in_existing_mixture = True
                if mixture['container'] == current_location:
                    if object_name in self.mixtures[idx_mix]['contents']:
                        self.mixtures[idx_mix]['contents'].remove(object_name)
                    else:
                        pass
                        # print(f"WARNING: {object_name} not found in mixture in {self.mixtures[idx_mix]}. Something's wrong!")
                    if len(self.mixtures[idx_mix]['contents']) == 0:
                        idx_mix_pop = idx_mix
                    else:
                        pass
                        # print(f"WARNING: Only {object_name}, and not all objects, were removed from mixture in {self.mixtures[idx_mix]}")
            if idx_mix_pop is not None:
                self.mixtures.pop(idx_mix_pop)
            if not added_in_existing_mixture:
                self.mixtures.append({'container':to_location, 'contents':[object_name]})
        self.full_scene[object_name]["location"] = to_location
        if object_name not in self.entities_created:
            if object_name not in self.objects_used:
                self.objects_used.append(object_name)
        if to_location not in self.objects_used:
            self.objects_used.append(to_location)
        if 'hot' in self.full_scene[to_location]['state'] or (to_location in list_cooking_appliances and 'on' in self.full_scene[to_location]['state']):
            self.full_scene[object_name]['state'].append('hot')
            self.objects_cooked.append(object_name)
        if to_location in list_of_dining_locations or self.full_scene[to_location]['location'] in list_of_dining_locations:
            self.serving_order.append(object_name)
            self.serving_order += self._get_contents(object_name, edible_only=True).split(', ')
        if any(["contains" in s for s in self.full_scene[object_name]['state']]):
            return (True, f"The whole container {obj_desc(object_name, self.full_scene[object_name])} is now in/on {to_location}. Be sure to use 'pour' or 'move_from' to extract contents ({self._get_contents(object_name, edible_only=True)}) before using them.")
        else:
            return (True, f"{obj_desc(object_name, self.full_scene[object_name])} is now in/on {to_location}.")
            
    def serve(self, arguments):
        return self.move(arguments)
    
    def place(self, arguments):
        return self.move(arguments)

    def _get_instance_id(self, given_name):
        given_name = given_name.replace(' ','_')
        if given_name in self.full_scene.keys():
            return given_name
        if '_' in given_name:
            if given_name[given_name.rfind('_')+1:].isdigit():
                return given_name
        for i in range(100):
            if (f'{given_name}_{i}' not in self.full_scene.keys()):
                return f'{given_name}_{i}'

    def move_from(self, arguments):
        object_name, from_location, to_location = arguments
        if from_location not in self.full_scene.keys():
            return (False, f"Action Failed! Location {from_location} does not exist")
        if object_name not in self.full_scene.keys():
            ## Create a new node 
            if f'contains {object_name}' not in self.full_scene[from_location]['state']:
                contents_str = self._get_contents(from_location)
                if len(contents_str) > 0:
                    contents_str = f" Did you mean one of {self._get_contents(from_location)}."
                else:
                    contents_str = f" {from_location} is empty."
                return (False, f"Action Failed! {obj_desc(from_location, self.full_scene[from_location])} does not contain {object_name}.{contents_str}")
            new_node = {"description": object_name, "state": [], "location":from_location, "type": ["content"]}
            object_name = self._deduplicate_name(object_name)
            self.actions_performed[-1] = (self.actions_performed[-1][0],(object_name, from_location, to_location))
            self.entities_created[object_name]=self.step_num
            self.full_scene[object_name] = new_node
            if object_name not in self.transformations[from_location]:
                self.transformations[from_location].append(object_name)
            if from_location not in self.objects_used:
                self.objects_used.append(from_location)
        else:
            if self.full_scene[object_name]['location'] != from_location:
                return (False, f"Action Failed! {object_name} is not in {from_location}. Did you mean Move {object_name} from {self.full_scene[object_name]['location']} to {to_location}?")
        return self.move([object_name, to_location])

    def _convert_contents_to_new_entity(self, container_name, result):
        past_contents = self._get_contents(container_name, edible_only=True)
        if len(past_contents) == 0:
            return (False, f"Action Failed! {container_name} is empty")
        self._remove_contents(container_name)
        new_node_id = self._deduplicate_name(result)
        self.actions_performed[-1] = (self.actions_performed[-1][0],(container_name, new_node_id))
        new_node = {"description": result, "state": [], "location":container_name, "type": ["content"]}
        self.full_scene[new_node_id] = new_node
        for mixture in self.mixtures:
            if container_name == mixture['container']:
                for content in past_contents.split(', '):
                    if content not in mixture['contents']:
                        if ('edible' in self.full_scene[content]['type'] or 'content' in self.full_scene[content]['type']):
                            mixture['contents'].append(content)
                mixture['contents'] = [new_node_id]
        for content in past_contents.split(', '):
            for original, avatars in self.transformations.items():
                if (content in avatars or content == original) and new_node_id not in self.transformations[original]:
                    self.transformations[original].append(new_node_id)
        if container_name not in self.objects_used:
            self.objects_used.append(container_name)
        self.entities_created[new_node_id]=self.step_num
        return (True, f"{past_contents} in {container_name} is now {new_node_id}")

    def mix(self, arguments):
        container_name, result = arguments
        found, msg = self._check_existence(container_name)
        if not found: 
            if '<X>' in msg:
                entity = msg.split('<X>')[1].strip()
                arguments[0] = entity
                msg = msg.split('<X>')[0] + self._compose_example('mix', arguments)
            return (False, msg)
        if 'container' not in self.full_scene[container_name]['type']:
            return (False, f"Action Failed! {container_name} is not a container that you can mix things in!")
        return self._convert_contents_to_new_entity(container_name, result)

    def cook(self, arguments):
        container_name, result = arguments
        found, msg = self._check_existence(container_name)
        if not found: 
            if '<X>' in msg:
                entity = msg.split('<X>')[1].strip()
                arguments[0] = entity
                msg = msg.split('<X>')[0] + self._compose_example('cook', arguments)
            return (False, msg)
        if self.full_scene[container_name]['location'] not in list_cooking_appliances and container_name not in list_cooking_appliances:
            return (False, f"Action Failed! {container_name} is on {self.full_scene[container_name]['location']}, which is not a cooking appliance.")
        if 'container' not in self.full_scene[container_name]['type']:
            return (False, f"Action Failed! {container_name} is not a container that you can cook things in!")
        if RELAXED_COOK_CHECK:
            self._set_cooked(container_name)
        return self._convert_contents_to_new_entity(container_name, result)

    def pour(self, arguments):
        if len(arguments) != 3:
            return (False, "Action Failed! Pour requires three arguments")
        entity, from_container, to_container = arguments
        return self.move_from([entity, from_container, to_container])

    def pour_into(self, arguments):
        return self.pour(arguments)

    def _convert_object_to_new_entity(self, action, object_name, result):
        if 'edible' not in self.full_scene[object_name]['type'] and 'content' not in self.full_scene[object_name]['type']:
            contents = self._get_contents(object_name, edible_only=True)
            msg = f"Action Failed! cannot {action} {object_name}, because {object_name} ({self.full_scene[object_name]['description']}) is not edible."
            if len(contents.split(', ')) > 0:
                if any(['contains' in s for s in self.full_scene[object_name]['state']]):
                    msg += f" Use 'pour' or 'move_from' to extract one of the contents ({contents}) first."
                else:
                    msg += f" Did you mean {action} items in {object_name}?"
            return (False, msg)
        result = self._deduplicate_name(result)
        self.actions_performed[-1] = (self.actions_performed[-1][0],(object_name, result))
        for mixture in self.mixtures:
            if object_name in mixture['contents']:
                mixture['contents'].remove(object_name)
                mixture['contents'].append(result)
        for original, avatars in self.transformations.items():
            if object_name in avatars and result not in self.transformations[original]:
                self.transformations[original].append(result)
            if object_name == original and result not in self.transformations[original]:
                self.transformations[original].append(result)
                if object_name not in self.objects_used:
                    self.objects_used.append(object_name)
        self.entities_created[result]=self.step_num
        self.full_scene[result] = deepcopy(self.full_scene[object_name])
        self.full_scene[result]['state'].append(f"{action}ed")
        del self.full_scene[object_name]
        return (True, f"{object_name} is now {result}")

    def chop(self, arguments):
        object_name, result = arguments
        found, msg = self._check_existence(object_name)
        if not found: 
            if '<X>' in msg:
                entity = msg.split('<X>')[1].strip()
                arguments[0] = entity
                msg = msg.split('<X>')[0] + self._compose_example('chop', arguments)
            return (False, msg)
        if self.full_scene[object_name]['location'] not in list_chopping_surfaces:
            return (False, f"Action Failed! {object_name} is on {self.full_scene[object_name]['location']}, which is not a valid chopping surface.")
        return self._convert_object_to_new_entity('chop', object_name, result)

    def chop_obj(self, arguments):
        return self.chop(arguments)

    def cook_entity_check(self, action_name, object_name=None, container_name=None):
        if container_name is None:
            container_name = self.full_scene[object_name]['location']
        else:
            assert object_name is None
        cooking_appliance_used = None
        if container_name == 'home':
            return (False, f"Action Failed! {container_name} is not on a cooking appliance.")
        if self.full_scene[container_name]['location'] not in list_cooking_appliances and container_name not in list_cooking_appliances:
            return (False, f"Action Failed! {container_name} is on {self.full_scene[container_name]['location']}, which is not a cooking appliance.")
        if RELAXED_COOK_CHECK:
            self._set_cooked(container_name)
        return (True, cooking_appliance_used)  

    def freeform(self, arguments):
        if len(arguments) != 3:
            return (False, "Action Failed! Freeform action requires two arguments")
        if  ' ' in arguments[1]:
            for argument in arguments[1]:
                if argument in self.get_foods():
                    suggestion = f"{arguments[0]} the object {arguments[1]} to get {arguments[2]}"
                    return (False, f"Action Failed! Freeform action should be a single word. Did you mean \'{suggestion}\'")
            for argument in arguments[1]:
                if argument in (self.get_containers() + self.get_food_packets()):
                    suggestion = f"{arguments[0]} items in {arguments[1]} to get {arguments[2]}"
                    return (False, f"Action Failed! Freeform action should be a single word. Did you mean \'{suggestion}\'")
            else:
                return (False, f"Action Failed! Freeform action requires the content to be food or something that contains food")
        action_name, object_name, result = arguments
        found, msg = self._check_existence(object_name)
        if not found: 
            if '<X>' in msg:
                entity = msg.split('<X>')[1].strip()
                arguments[0] = entity
                msg = msg.split('<X>')[1] + self._compose_example('freeform', arguments)
            return (False, msg)
        cooking_actions = ['heat', 'cook', 'simmer', 'fry', 'saute', 'boil', 'poach', 'toast', 'bake']
        if action_name in cooking_actions:
            okay, msg = self.cook_entity_check(action_name, object_name=object_name)
            if not okay:
                return (False, msg)
        return self._convert_object_to_new_entity(action_name, object_name, result)

    def freeform_contents(self, arguments):
        if len(arguments) != 3:
            return (False, "Action Failed! Freeform for contents action requires two arguments")
        action_name, container_name, result = arguments
        if  ' ' in arguments[1]:
            for argument in arguments[1]:
                if argument in self.get_foods():
                    suggestion = f"{arguments[0]} the object {arguments[1]} to get {arguments[2]}"
                    return (False, f"Action Failed! Freeform action should be a single word. Did you mean \'{suggestion}\'")
            for argument in arguments[1]:
                if argument in (self.get_containers() + self.get_food_packets()):
                    suggestion = f"{arguments[0]} items in {arguments[1]} to get {arguments[2]}"
                    return (False, f"Action Failed! Freeform action should be a single word. Did you mean \'{suggestion}\'")
            else:
                return (False, f"Action Failed! Freeform action requires the content to be food or something that contains food")
        found, msg = self._check_existence(container_name)
        if not found: 
            if '<X>' in msg:
                entity = msg.split('<X>')[1].strip()
                arguments[0] = entity
                msg = msg.split('<X>')[1] + self._compose_example('freeform', arguments)
            return (False, msg)
        if 'container' not in self.full_scene[container_name]['type']:
            return (False, f"Action Failed! {container_name} is not a container that you can '{action_name}' things in!")
        cooking_actions = ['heat', 'cook', 'simmer', 'fry', 'saute', 'boil', 'poach', 'toast', 'bake']
        if action_name in cooking_actions:
            okay, msg = self.cook_entity_check(action_name, container_name=container_name)
            if not okay:
                return (False, msg)
        return self._convert_contents_to_new_entity(container_name, result)
    
    def ask(self, arguments):
        if len(arguments) != 1:
            return (False, "Action Failed! Ask requires one argument")
        question = arguments[0]
        return (True, "")

    def done(self, arguments):
        return (True, "DONE")

    def complete(self, arguments):
        return (True, "DONE")

    def undo(self, arguments):
        if self.step_num == 0:
            return (False, "Action Failed! Nothing to undo.")
        self.step_num -= 2
        self.restore_state()
        return (True, "Undone the last action.")

    def parse_and_execute_best(self, action_string):
        first_msg_action = None
        if ',' in action_string and 'ask' not in action_string.lower():
            return (False, f"Action Failed! Only one action can be done at a time. Action asks for {action_string.count(',')+1}.", None, None)
        for action, action_args in parse_action(action_string):
            action_func = getattr(self, action)
            if 'freeform' in action:
                self.actions_performed.append((action_args[0],action_args[1:]))
            else:
                self.actions_performed.append((action,action_args))
            success, msg = action_func(action_args)
            if success: 
                return success, msg, action, action_args
            if first_msg_action is None: first_msg_action = False, msg, action, action_args
        if first_msg_action is None:
            return (False, f"Action Failed! Unknown action {action_string}. Please choose from the provided list of actions (Open, Close, Search, Move, Mix, Cook, Chop, Pour, Ask, etc.), pay attention to the provided action templates, and say 'Declare Done' if you think the task is complete.", None, None)
        return first_msg_action

    def cache_state(self):
        self.prev_action_buffer = deepcopy(self.action_buffer)
        self.prev_actions_performed = deepcopy(self.actions_performed)
        self.prev_entities_created = deepcopy(self.entities_created)
        self.prev_objects_used = deepcopy(self.objects_used)
        self.prev_mixtures = deepcopy(self.mixtures)
        self.prev_transformations = deepcopy(self.transformations)
        self.prev_state = deepcopy(self.full_scene)

    def restore_state(self):
        self.action_buffer = deepcopy(self.prev_action_buffer)
        self.actions_performed = deepcopy(self.prev_actions_performed)
        self.entities_created = deepcopy(self.prev_entities_created)
        self.objects_used = deepcopy(self.prev_objects_used)
        self.mixtures = deepcopy(self.prev_mixtures)
        self.transformations = deepcopy(self.prev_transformations)
        self.full_scene = deepcopy(self.prev_state)

    def step(self, action_string):
        self.action_buffer.append(action_string)
        global NO_REPEAT_HORIZON, NO_REPEAT_MAX_REPITITIONS
        if self.action_buffer[-NO_REPEAT_HORIZON-1:-1].count(action_string) >= NO_REPEAT_MAX_REPITITIONS:
            self.actions_performed = self.actions_performed[:self.step_num]
            self.invalid_actions[action_string] = self.step_num
            return (False, f"Action Failed! Repeated action {action_string} too many times. Please choose a different action.", None, None)
        success, msg, action, action_args = self.parse_and_execute_best(action_string)
        if not success:
            self.actions_performed = self.actions_performed[:self.step_num]
            return success, msg, action, action_args
        self.step_num += 1
        self.cache_state()
        return success, msg, action, action_args

    def get_foods(self):
        return [k for k,desc in self.full_scene.items() if 'edible' in desc["type"] or 'content' in desc["type"]]

    def get_containers(self, filled_only=False):
        all_containers = [k for k,desc in self.full_scene.items() if 'container' in desc["type"]]
        filled_containers = []
        if filled_only:
            for container in all_containers:
                if len(self._get_contents(container, edible_only=True)) > 0:
                    filled_containers.append(container)
        return filled_containers if filled_only else all_containers

    def get_food_packets(self):
        return [k for k,desc in self.full_scene.items() if any(['contains' in state for state in desc["state"]])]

    def get_all_entities(self, including_contents=False):
        entities = set(self.full_scene.keys())
        if including_contents:
            for obj in self.full_scene.keys():
                entities.update((self._get_contents(obj)).split(', '))
        return list(entities)

    def get_content_from_container(self):
        phrases = []
        for obj in self.full_scene.keys():
            contents = (self._get_contents(obj, edible_only=True)).split(', ')
            for content in contents:
                content = content.strip()
                if content != '':
                    phrases.append(f"{content} from {obj}")
        return phrases
    
    def summarize_progress(self):
        entities_created_that_still_exist = [entity for entity in self.entities_created if entity in self.full_scene.keys()]
        if len(entities_created_that_still_exist) == 0:
            return "You have not made anything yet."
        summary = ''
        summary += 'So far, you have made '
        for entity in entities_created_that_still_exist:
            sources = [source for source, objects in self.transformations.items() if entity in objects]
            summary += f"{entity} in/on {self.full_scene[entity]['location']} from {', '.join(sources)}, "
        summary = summary[:-2] + '.'
        return summary

    def get_cookable_entities(self):
        entitites_on_cooking_appliances = []
        foods = self.get_foods()
        for entity in self.full_scene.keys():
            if self.full_scene[entity]['location'] in list_cooking_appliances and entity in foods:
                entitites_on_cooking_appliances.append(entity)
        return entitites_on_cooking_appliances

    def get_choppable_entities(self):
        entitites_on_chopping_surfaces = []
        foods = self.get_foods()
        for entity in self.full_scene.keys():
            if self.full_scene[entity]['location'] in list_chopping_surfaces and entity in foods:
                entitites_on_chopping_surfaces.append(entity)
        return entitites_on_chopping_surfaces

    def get_full_state(self):
        for mixture in self.mixtures:
            sources = []
            for content in mixture['contents']:
                sources += [source for source, objects in self.transformations.items() if content in objects or content == source]
            mixture['sources'] = sources
        state = {}
        def any_transf_of_obj_is_cooked(obj):
            if obj not in self.transformations: 
                return False
            for o in self.transformations[obj]:
                if o in self.objects_cooked:
                    return True
            return False
        state["objects_cooked"] = [obj for obj in self.transformations if (obj in self.objects_cooked or any_transf_of_obj_is_cooked(obj))]
        state["serving_order"] = []
        for served in self.serving_order:
            for obj, transf_list in self.transformations.items():
                if served in transf_list:
                    state["serving_order"].append(obj)
            state["serving_order"].append(served)
        state["actions_performed"] = self.actions_performed
        state["entities_created"] = self.entities_created
        state["entities_created_that_remain"] = [e for e in self.entities_created if e in self.full_scene.keys()]
        state["objects_used"] = list(self.objects_used)
        state["mixtures"] = self.mixtures
        state["transformations"] = self.transformations
        state["final_object_locations"] = {k: v["location"] for k, v in self.full_scene.items()}
        return state
