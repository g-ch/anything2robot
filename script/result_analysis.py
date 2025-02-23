import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
import os
import sys
from progress.bar import IncrementalBar

file_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(file_path, '../auto_design'))
sys.path.append(os.path.join(file_path, '../auto_design/modules'))

from interference_removal import RobotOptResult

'''
This class is used to parse the result of one round of optimization. Mainly parse the log.pkl file and log.txt file.
Check Useful Attributes to see what attributes are available.
Some attributes are not available in all rounds, the result will be None if the attribute is not available.
'''
class ResultOneRound:
    def __init__(self, round_folder_path):

        ### USEFUL ATTRIBUTES ###
        self.round_folder_path = round_folder_path
        self.log_pkl_path = None
        self.model_name = self.round = self.exit_code = None
        self.decompose_voxel_num = self.decompose_time = None
        self.motor_opt_cost_log = self.motor_opt_time = None
        self.joint_connect_voxel_num = self.joint_connect_time = None
        self.interference_removal_voxel_num = self.interference_removal_time = None
        self.result_saving_time = self.destruction_check_time = None
        self.mapdl_connection_failure = self.mapdl_error = False

        # Find a pkl file in the folder that contains "round"
        for file in os.listdir(round_folder_path):
            if file.endswith('.pkl') and "round" in file:
                self.log_pkl_path = os.path.join(round_folder_path, file)
                break
        
        if self.log_pkl_path is None:
            print("No pkl file found in the folder")
            return
        else:
            self.log_dict = self.decode_log_pkl(self.log_pkl_path)
            self.model_name = self.log_dict['model_name']
            self.round = self.log_dict['round']
            self.exit_code = self.log_dict['exit_code']

            self.decompose_voxel_num = self.try_to_get_item_from_log_dict('decompose_voxel_num')
            self.decompose_time = self.try_to_get_item_from_log_dict('decompose_time')
            self.motor_opt_cost_log = self.try_to_get_item_from_log_dict('motor_opt_cost_log')  # Motor cost log: cost_motor_position, cost_motor_occupancy, two_degree_rotation_interference_cost
            self.motor_opt_time = self.try_to_get_item_from_log_dict('motor_opt_time')
            self.joint_connect_voxel_num = self.try_to_get_item_from_log_dict('joint_connect_voxel_num')
            self.joint_connect_time = self.try_to_get_item_from_log_dict('joint_connect_time')
            self.interference_removal_voxel_num = self.try_to_get_item_from_log_dict('interference_removal_voxel_num')
            self.interference_removal_time = self.try_to_get_item_from_log_dict('interference_removal_time')
            self.result_saving_time = self.try_to_get_item_from_log_dict('result_saving_time')
            self.destruction_check_time = self.try_to_get_item_from_log_dict('destruction_check_time')
            self.fea_time = self.try_to_get_item_from_log_dict('fea_time')

        # Find a txt file in the folder that contains "round"
        for file in os.listdir(round_folder_path):
            if file.endswith('.txt'):
                self.log_txt_path = os.path.join(round_folder_path, file)
                break
        
        if self.log_txt_path is None:
            print("No txt file found in the folder")
            return
        else:
            # Check if "MAPDL server connection terminated" is in the txt file
            with open(self.log_txt_path, 'r') as f:
                if "MAPDL server connection terminated" in f.read():
                    self.mapdl_connection_failure = True
                if "*** ERROR ***" in f.read():
                    self.mapdl_error = True

    def try_to_get_item_from_log_dict(self, item_name, default_value=None):
        try:
            return self.log_dict[item_name]
        except KeyError:
            return default_value

    def decode_log_pkl(self, log_pkl_path):
        with open(log_pkl_path, 'rb') as f:
            log_dict = pkl.load(f)
        
        return log_dict

'''
This class is used to parse the results of multiple rounds of optimization. Mainly parse the results of all rounds in a folder.
Check Useful Attributes to see what attributes are available.
'''
class ResultMultipleRounds:
    def __init__(self, folder_path):
        ### USEFUL ATTRIBUTES ###
        self.folder_path = folder_path
        self.result_all_rounds = []
        self.round_num = None
        self.model_name = self.valid_flag = self.success_flag = None
        self.success_round_id = None
        self.success_round_data = None
        self.failure_codes = None

        # Find all folders in the folder that contains "round"
        for folder in os.listdir(folder_path):
            if os.path.isdir(os.path.join(folder_path, folder)) and "round" in folder:
                result_this_round = ResultOneRound(os.path.join(folder_path, folder))
                if result_this_round.log_pkl_path is not None:
                    self.result_all_rounds.append(result_this_round)

        # Check round number
        self.round_num = len(self.result_all_rounds)
        if self.round_num == 0:
            print("No round found in the folder")
            return
        
        # Sort the results by round number
        self.result_all_rounds.sort(key=lambda x: x.round)
        for i, result in enumerate(self.result_all_rounds):
            if result.round != i+1:
                print(f"Round {i+1} is missing")
                break
        
        self.model_name = self.result_all_rounds[0].model_name
        self.valid_flag = True
        self.success_flag = False
        self.success_round_data = None
        self.failure_codes = self.get_failure_codes()
        
        # Check if operation is invalid because of MAPDL failure
        if self.check_mapdl_failure():
            self.valid_flag = False
        else:
            # Check if there is a successful round
            for result in self.result_all_rounds:
                if result.exit_code == 0:
                    self.success_flag = True
                    self.success_round_id = result.round
                    self.success_round_data = result
                    break
        

    def check_mapdl_failure(self):
        for result in self.result_all_rounds:
            if result.mapdl_connection_failure or result.mapdl_error:
                print(f"Round {result.round}: MAPDL failure")
                return True
        return False

    def get_failure_codes(self):
        failure_codes = []
        for result in self.result_all_rounds:
            if result.exit_code != 0:
                failure_codes.append(result.exit_code)
        return failure_codes

'''
This class is used to parse the results of multiple models in one folder.
Check Useful Attributes to see what attributes are available.
Use get_success_rate() to get the success rate of all models.
'''
class DatasetResultAnalysis:
    def __init__(self, result_folder):
        
        ### USEFUL ATTRIBUTES ###
        self.model_results = []

        # Find all subfolders in the result folder and parse the results of each model
        sub_folders = os.listdir(result_folder)
        for folder in sub_folders:
            if os.path.isdir(os.path.join(result_folder, folder)):
                result = ResultMultipleRounds(os.path.join(result_folder, folder))
                if result.round_num > 0:
                    self.model_results.append(result)

        self.merge_results() # Merge results with the same model name

    '''
    This function is used to merge results with the same model name.
    Used in initialize the DatasetResultAnalysis class.
    '''
    def merge_results(self):
        # If two models have the same name, merge their results. If result is successful, keep the successful one. If both are failed, keep the valid one. If both are invalid, keep the one with more rounds.
        model_name_dict = {}
        for model_result in self.model_results:
            name_this = model_result.model_name
            if name_this in model_name_dict:
                model_result_previous = model_name_dict[name_this]
                if model_result_previous.success_flag:
                    continue
                elif model_result.success_flag:
                    model_name_dict[name_this] = model_result
                else:
                    if model_result_previous.valid_flag:
                        if model_result.valid_flag:
                            if model_result_previous.round_num < model_result.round_num:
                                model_name_dict[name_this] = model_result
                    else:
                        if model_result.valid_flag:
                            model_name_dict[name_this] = model_result
                print(f"Model {name_this} has multiple results. Merged.")
            else:
                model_name_dict[name_this] = model_result

        self.model_results = list(model_name_dict.values())

    '''
    This function is used to get the time consumption of all rounds. Check return value for more details.
    max_round_num is the maximum number of rounds to consider.
    '''
    def get_time_consumption(self, max_round_num=8):        
        total_time_consumption_rounds = np.zeros(max_round_num)
        max_time_consumption_rounds = np.zeros(max_round_num)
        min_time_consumption_rounds = np.ones(max_round_num) * 1e10
        number_of_samples_each_round = np.zeros(max_round_num)

        total_success_result_num = 0
        for model_result in self.model_results:
            if not model_result.valid_flag:
                continue

            max_round_id = model_result.success_round_id
            if max_round_id is None:
                max_round_id = model_result.round_num
            else:
                total_success_result_num += 1
            
            for i in range(max_round_id):
                #time_this_round = model_result.result_all_rounds[i].decompose_time + model_result.result_all_rounds[i].motor_opt_time + model_result.result_all_rounds[i].joint_connect_time + model_result.result_all_rounds[i].interference_removal_time + model_result.result_all_rounds[i].result_saving_time + model_result.result_all_rounds[i].destruction_check_time + model_result.result_all_rounds[i].fea_time
                
                time_this_round = 0
                time_this_round = self.add_if_not_none(time_this_round, model_result.result_all_rounds[i].decompose_time)
                time_this_round = self.add_if_not_none(time_this_round, model_result.result_all_rounds[i].motor_opt_time)
                time_this_round = self.add_if_not_none(time_this_round, model_result.result_all_rounds[i].joint_connect_time)
                time_this_round = self.add_if_not_none(time_this_round, model_result.result_all_rounds[i].interference_removal_time)
                time_this_round = self.add_if_not_none(time_this_round, model_result.result_all_rounds[i].result_saving_time)
                time_this_round = self.add_if_not_none(time_this_round, model_result.result_all_rounds[i].destruction_check_time)
                time_this_round = self.add_if_not_none(time_this_round, model_result.result_all_rounds[i].fea_time)
                
                total_time_consumption_rounds[i] += time_this_round
                if time_this_round > max_time_consumption_rounds[i]:
                    max_time_consumption_rounds[i] = time_this_round
                if time_this_round < min_time_consumption_rounds[i]:
                    min_time_consumption_rounds[i] = time_this_round
                number_of_samples_each_round[i] += 1
        
        avg_time_consumption_rounds = total_time_consumption_rounds / number_of_samples_each_round
        print(f"Average time consumption rounds: {avg_time_consumption_rounds}")
        print(f"Max time consumption rounds: {max_time_consumption_rounds}")
        print(f"Min time consumption rounds: {min_time_consumption_rounds}")

        time_consumption_each_success_part = np.zeros((total_success_result_num, 5))
        seq = 0
        for model_result in self.model_results:
            if not model_result.valid_flag:
                continue
            if not model_result.success_flag:
                continue
            
            time_consumption_each_success_part[seq, 0] = model_result.success_round_data.decompose_time
            time_consumption_each_success_part[seq, 1] = model_result.success_round_data.motor_opt_time
            time_consumption_each_success_part[seq, 2] = model_result.success_round_data.joint_connect_time
            time_consumption_each_success_part[seq, 3] = model_result.success_round_data.interference_removal_time + model_result.success_round_data.result_saving_time + model_result.success_round_data.destruction_check_time
            time_consumption_each_success_part[seq, 4] = model_result.success_round_data.fea_time
            
            seq += 1
            if seq > total_success_result_num:
                print("Error: seq > total_success_result_num")
                break
        
        return avg_time_consumption_rounds, max_time_consumption_rounds, min_time_consumption_rounds, time_consumption_each_success_part
    
    '''
    This function is used to add two numbers if they are not None.
    '''
    def add_if_not_none(self, a, b):
        if a is not None:
            if b is not None:
                return a + b
            else:
                return a
        else:
            return b
        
    '''
    This function is used to get the success rate of all models.
    log_csv_path is the path to save the log if provided.
    max_round_num is the maximum number of rounds to consider.
    '''
    def get_success_rate(self, log_csv_path=None, max_round_num=8):
        valid_num = 0
        success_num = 0
        failure_code_1_num = 0
        failure_code_2_num = 0
        failure_code_3_num = 0
        failure_code_4_num = 0
        failure_code_others = 0

        failure_codes_round = np.zeros((max_round_num, 5)) # 4 failure codes: 1, 2, 3, 4, (5 is others)
        growing_success_rate_rounds = np.ones(max_round_num)

        log_dict = {}
        for model_result in self.model_results:
            if model_result.valid_flag:
                valid_num += 1
                if model_result.success_flag:
                    success_num += 1
                
                round = 0
                for code in model_result.failure_codes: 
                    if code == 1:
                        failure_code_1_num += 1
                    elif code == 2:
                        failure_code_2_num += 1
                    elif code == 3:
                        failure_code_3_num += 1
                    elif code == 4:
                        failure_code_4_num += 1
                    else:
                        failure_code_others += 1
                        code = 5 # others

                    failure_codes_round[round, code-1] += 1
                    round += 1

            log_dict[model_result.model_name] = [model_result.valid_flag, model_result.success_flag, model_result.success_round_id, model_result.failure_codes]

        for i in range(max_round_num):
            if valid_num > 0:
                growing_success_rate_rounds[i] = (valid_num - failure_codes_round[i].sum()) / valid_num

        # Save the log to a csv file if log_csv_path is provided
        if log_csv_path is not None:
            if log_csv_path.endswith('.csv'):
                with open(log_csv_path, 'w') as f:
                    f.write('model_name,valid_flag,success_flag,success_round,failure_codes\n')
                    for model_name in log_dict:
                        valid_flag, success_flag, success_round, failure_codes = log_dict[model_name]
                        failure_codes_str = ','.join([str(code) for code in failure_codes])
                        f.write(f"{model_name},{valid_flag},{success_flag},{success_round},{failure_codes_str}\n")
            else:
                print("Invalid csv path")
        
        valid_rate = valid_num / len(self.model_results)
        success_rate = success_num / valid_num
        print(f"Total num: {len(self.model_results)}")
        print(f"Valid num: {valid_num}")
        print(f"Valid rate: {valid_rate}")
        print(f"Success rate: {success_rate}")
        # print(f"Failure code 1 num: {failure_code_1_num} The motor cost is too high.")
        # print(f"Failure code 2 num: {failure_code_2_num} The mesh is destroyed after interference removal.")
        # print(f"Failure code 3 num: {failure_code_3_num} The mesh is not watertight after interference removal.")
        # print(f"Failure code 4 num: {failure_code_4_num} The mesh is not feasible in FEA.")
        # print(f"Failure codes round: {failure_codes_round}")
        print(f"Success rate round: {growing_success_rate_rounds}")

        failure_codes_num = [failure_code_1_num, failure_code_2_num, failure_code_3_num, failure_code_4_num]

        return valid_rate, success_rate, failure_codes_num, failure_codes_round, growing_success_rate_rounds

    '''
    This function is used to get the motor cost of all rounds. The result is a map of round id to a list of motor costs of each success round.
    max_round_num is the maximum number of rounds to consider.
    '''
    def get_motor_cost(self, max_round_num=8):
        motor_position_cost_rounds_map = {}
        motor_occupancy_cost_rounds_map = {}
        two_degree_rotation_interference_cost_rounds_map = {}

        for i in range(1, max_round_num+1):
            motor_position_cost_rounds_map[i] = []
            motor_occupancy_cost_rounds_map[i] = []
            two_degree_rotation_interference_cost_rounds_map[i] = []

        for model_result in self.model_results:
            if not model_result.valid_flag:
                continue
            if not model_result.success_flag:
                continue
            
            success_round_data = model_result.success_round_data
            success_round_id = model_result.success_round_id
            print(f"Model {model_result.model_name} success round id: {success_round_id}")

            if success_round_id is None:
                print(f"Model {model_result.model_name} has no success round id")
                continue
            
            motor_cost_log = success_round_data.motor_opt_cost_log
            if motor_cost_log is None:
                print(f"Model {model_result.model_name} has no motor cost log")
                continue
            else:
                print(f"****** {model_result.model_name} ******")
                # print(motor_cost_log)

                # Convert to numpy array for easier computation
                cost_array = np.array(motor_cost_log)
                # Sum along axis 1 to get total cost for each iteration
                total_costs = np.sum(cost_array, axis=1)
                min_cost = np.min(total_costs)
                min_cost_idx = np.argmin(total_costs)
                print(f"Minimum cost: {min_cost}")
                print(f"Minimum cost index: {min_cost_idx}")

                motor_position_cost_rounds_map[success_round_id].append(cost_array[min_cost_idx, 0])
                motor_occupancy_cost_rounds_map[success_round_id].append(cost_array[min_cost_idx, 1])
                two_degree_rotation_interference_cost_rounds_map[success_round_id].append(cost_array[min_cost_idx, 2])

        return motor_position_cost_rounds_map, motor_occupancy_cost_rounds_map, two_degree_rotation_interference_cost_rounds_map
            


    def get_mesh_similarity(self, max_round_num=8):
        hausdorff_distance_rounds_map = {}
        average_point_distance_rounds_map = {}

        # Initialize the maps
        for i in range(1, max_round_num+1):
            hausdorff_distance_rounds_map[i] = []
            average_point_distance_rounds_map[i] = []

        # Get the mesh similarity
        bar = IncrementalBar('Processing', max=len(self.model_results))
        for model_result in self.model_results:
            bar.next()
            if not model_result.valid_flag:
                continue
            if not model_result.success_flag:
                continue
            
            success_round_id = model_result.success_round_id
            if success_round_id is None:
                print(f"Model {model_result.model_name} has no success round id")
                continue

            result_this_success_round = model_result.success_round_data
            folder = result_this_success_round.round_folder_path
            pkl_path = os.path.join(folder, 'robot_result.pkl')
            #Search for the stl file in the folder
            stl_path = None
            for file in os.listdir(folder):
                if file.endswith('.stl'):
                    stl_path = os.path.join(folder, file)
                    break
            if stl_path is None:
                print(f"Model {model_result.model_name} has no stl file")
                continue
            
            robot_result = pkl.load(open(pkl_path, 'rb'))
            hausdorff_distance, average_point_distance = robot_result.getMeshSimilarity(stl_dir=stl_path)
            # print(f"Hausdorff distance: {hausdorff_distance}")
            # print(f"Average point distance: {average_point_distance}")

            hausdorff_distance_rounds_map[success_round_id].append(hausdorff_distance)
            average_point_distance_rounds_map[success_round_id].append(average_point_distance)
            bar.finish()

        return hausdorff_distance_rounds_map, average_point_distance_rounds_map
    

if __name__ == '__main__':
    # round_folder_path = '/media/clarence/Clarence/anything2robot/result/n02085782_2100_neutral_res_e300_smoothed_scaled_20241028-063017/result_round1'
    # round_result = ResultOneRound(round_folder_path)
    # print(round_result.model_name)
    # print(round_result.round)
    # print(round_result.exit_code)
    # print(round_result.log_dict)

    # model_folder_path = '/media/clarence/Clarence/anything2robot/result/n02085782_2100_neutral_res_e300_smoothed_scaled_20241028-063017'
    # model_result = ResultMultipleRounds(model_folder_path)
    # print(model_result.valid_flag)
    # print(model_result.success_flag)

    # dataset_result_folder = '/home/cc/git/anything2robot/result'
    dataset_result_folder = "/media/clarence/Clarence/anything2robot_data/result"
    dataset_result_analysis = DatasetResultAnalysis(dataset_result_folder)
    max_round_num = 8

    ########### Motor cost ###########
    motor_position_cost_rounds_map, motor_occupancy_cost_rounds_map, __ = dataset_result_analysis.get_motor_cost(max_round_num=max_round_num)

    # Plot the average motor position cost and motor occupancy cost of each round in a bar plot. Add the     
    motor_position_cost_rounds_avg = np.zeros(max_round_num)
    motor_occupancy_cost_rounds_avg = np.zeros(max_round_num)
    motor_position_cost_rounds_std = np.zeros(max_round_num)
    motor_occupancy_cost_rounds_std = np.zeros(max_round_num)
    for round_id in range(1, max_round_num+1):
        # Turn the list into a numpy array
        motor_position_cost_this_round = np.array(motor_position_cost_rounds_map[round_id])
        motor_occupancy_cost_this_round = np.array(motor_occupancy_cost_rounds_map[round_id])
        motor_position_cost_rounds_avg[round_id-1] = motor_position_cost_this_round.mean()
        motor_occupancy_cost_rounds_avg[round_id-1] = motor_occupancy_cost_this_round.mean()
        motor_position_cost_rounds_std[round_id-1] = motor_position_cost_this_round.std()
        motor_occupancy_cost_rounds_std[round_id-1] = motor_occupancy_cost_this_round.std()

    # Draw a bar plot of motor_position_cost_rounds_avg and motor_occupancy_cost_rounds_avg
    x = np.arange(1, max_round_num + 1)
    width = 0.35  # Width of the bars

    # Create the bar plot
    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width/2, motor_position_cost_rounds_avg, width, 
                    yerr=motor_position_cost_rounds_std, 
                    label='Position Cost',
                    capsize=5)
    rects2 = ax.bar(x + width/2, motor_occupancy_cost_rounds_avg, width, 
                    yerr=motor_occupancy_cost_rounds_std,
                    label='Occupancy Cost',
                    capsize=5)

    # Customize the plot
    ax.set_xlabel('Round Number')
    ax.set_ylabel('Cost')
    ax.set_title('Position and Occupancy Costs per Round')
    ax.set_xticks(x)
    ax.set_ylim(0, 5)  # Set y-axis limits from 0 to 5
    ax.legend()

    # Add some padding to the y-axis to make error bars fully visible
    ax.margins(y=0.1)
    # Add grid for better readability
    ax.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.show()


    ########### Mesh cost ###########
    hausdorff_distance_pkl_path = dataset_result_folder + '/hausdorff_distance_rounds_map.pkl'
    average_point_distance_pkl_path = dataset_result_folder + '/average_point_distance_rounds_map.pkl'
    # If the pkl file exists, load it
    if os.path.exists(hausdorff_distance_pkl_path) and os.path.exists(average_point_distance_pkl_path):
        print(f"Loading hausdorff_distance_rounds_map and average_point_distance_rounds_map from {hausdorff_distance_pkl_path} and {average_point_distance_pkl_path}")
        with open(hausdorff_distance_pkl_path, 'rb') as f:
            hausdorff_distance_rounds_map = pkl.load(f)
        with open(average_point_distance_pkl_path, 'rb') as f:
            average_point_distance_rounds_map = pkl.load(f)
    else:
        hausdorff_distance_rounds_map, average_point_distance_rounds_map = dataset_result_analysis.get_mesh_similarity(max_round_num=max_round_num)
        # Save to a binary file
        with open(hausdorff_distance_pkl_path, 'wb') as f:
            pkl.dump(hausdorff_distance_rounds_map, f)
        with open(average_point_distance_pkl_path, 'wb') as f:
            pkl.dump(average_point_distance_rounds_map, f)

    # Find the average and std of the average point distance and hausdorff distance of each round
    scale_factor = 1.0 / 1.1
    average_point_distance_rounds_avg = np.zeros(max_round_num)
    average_point_distance_rounds_std = np.zeros(max_round_num)
    scaled_average_point_distance_rounds_avg = np.zeros(max_round_num)
    hausdorff_distance_rounds_avg = np.zeros(max_round_num)
    hausdorff_distance_rounds_std = np.zeros(max_round_num)
    scaled_hausdorff_distance_rounds_avg = np.zeros(max_round_num)
    for round_id in range(1, max_round_num+1):
        average_point_distance_this_round = np.array(average_point_distance_rounds_map[round_id])
        average_point_distance_rounds_avg[round_id-1] = average_point_distance_this_round.mean()
        average_point_distance_rounds_std[round_id-1] = average_point_distance_this_round.std()
        hausdorff_distance_this_round = np.array(hausdorff_distance_rounds_map[round_id])
        hausdorff_distance_rounds_avg[round_id-1] = hausdorff_distance_this_round.mean()
        hausdorff_distance_rounds_std[round_id-1] = hausdorff_distance_this_round.std()

        # Scale the average point distance and hausdorff distance
        scaled_average_point_distance_rounds_avg[round_id-1] = average_point_distance_rounds_avg[round_id-1] * pow(scale_factor, round_id-1)
        scaled_hausdorff_distance_rounds_avg[round_id-1] = hausdorff_distance_rounds_avg[round_id-1] * pow(scale_factor, round_id-1)

    # Draw a bar plot of the average_point_distance_rounds_map and hausdorff_distance_rounds_map
    x = np.arange(1, max_round_num + 1)
    width = 0.35  # Width of the bars

    # Create the bar plot
    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x, average_point_distance_rounds_avg, width, 
                    yerr=average_point_distance_rounds_std, 
                    label='Average point distance',
                    capsize=5)
    
    ## Plot both average point distance and hausdorff distance in the same plot
    # rects1 = ax.bar(x - width/2, average_point_distance_rounds_avg, width, 
    #                 yerr=average_point_distance_rounds_std, 
    #                 label='Average point distance',
    #                 capsize=5)
    # rects2 = ax.bar(x + width/2, hausdorff_distance_rounds_avg, width, 
    #                 yerr=hausdorff_distance_rounds_std, 
    #                 label='Hausdorff distance',
    #                 capsize=5)

    # Add a line plot of the scaled average point distance and scaled hausdorff distance
    ax.plot(x, scaled_average_point_distance_rounds_avg, label='Scaled average point distance', color='r')
    # ax.plot(x, scaled_hausdorff_distance_rounds_avg, label='Scaled hausdorff distance', color='g')
    
    ax.set_xlabel('Round Number')
    ax.set_ylabel('Distance')
    ax.set_title('Mesh similarity per round')
    ax.set_xticks(x)
    ax.legend()
    
    # Add some padding to the y-axis to make error bars fully visible
    ax.margins(y=0.1)
    # Add grid for better readability
    ax.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.show()

    exit()

    ########### Success rate and time consumption ###########
    valid_rate, success_rate, failure_codes_num, failure_codes_round, growing_success_rate_rounds = dataset_result_analysis.get_success_rate(log_csv_path = dataset_result_folder + '/result_log.csv')
 
    avg_time_consumption_rounds, max_time_consumption_rounds, min_time_consumption_rounds, time_consumption_each_success_part = dataset_result_analysis.get_time_consumption()
    
    avg_time_each_success_part = np.mean(time_consumption_each_success_part, axis=0)
    std_dev_time_each_success_part = np.std(time_consumption_each_success_part, axis=0)
    print(f"Average time consumption each success part: {avg_time_each_success_part}")
    print(f"Std dev time consumption each success part: {std_dev_time_each_success_part}")

    growing_avg_time_consumption_rounds = np.zeros(growing_success_rate_rounds.shape)
    growing_max_time_consumption_rounds = np.zeros(growing_success_rate_rounds.shape)
    growing_min_time_consumption_rounds = np.zeros(growing_success_rate_rounds.shape)
    for i in range(growing_avg_time_consumption_rounds.shape[0]):
        for j in range(i+1):
            growing_avg_time_consumption_rounds[i] += avg_time_consumption_rounds[j]
            growing_max_time_consumption_rounds[i] += max_time_consumption_rounds[j]
            growing_min_time_consumption_rounds[i] += min_time_consumption_rounds[j]

    # Draw a bar plot of growing_success_rate_rounds and a line plot of growing_avg_time_consumption_rounds, growing_max_time_consumption_rounds, growing_min_time_consumption_rounds
    round_num = growing_success_rate_rounds.shape[0]
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.bar(np.arange(round_num), growing_success_rate_rounds, color='b', alpha=0.5)
    ax2.plot(np.arange(round_num), growing_avg_time_consumption_rounds, marker='o', color='r', label='Average time consumption')
    ax2.plot(np.arange(round_num), growing_max_time_consumption_rounds, marker='s', color='g', label='Max time consumption')
    ax2.plot(np.arange(round_num), growing_min_time_consumption_rounds,  marker='^', color='orange', label='Min time consumption')
    
    ax1.set_xlabel('Round')
    ax1.set_ylabel('Success rate', color='b')
    ax2.set_ylabel('Average time consumption', color='r')
    ax2.legend()
    plt.show()

    # Plot the time consumption of each success part in a box plot
    fig, ax = plt.subplots()
    ax.boxplot(time_consumption_each_success_part, patch_artist=True)
    ax.set_xticklabels(['Decompose', 'Motor opt', 'Joint connect', 'Interference removal', 'FEA'])
    ax.set_ylabel('Time consumption (s)')
    # Add grid
    ax.yaxis.grid(True)
    ax.xaxis.grid(True)
    plt.show()

    # Plot the average time consumption of each success part in a pie chart
    labels = ['Decompose', 'Motor opt', 'Joint connect', 'Interference removal', 'FEA']
    sizes = avg_time_each_success_part
    fig1, ax1 = plt.subplots()
    ax1.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
    ax1.axis('equal')
    plt.show()

    # Plot the reason of failure in a pie chart
    labels = ['Motor cost too high', 'Mesh destroyed in interference removal', 'FEA not feasible']
    sizes = [failure_codes_num[0], failure_codes_num[1]+failure_codes_num[2], failure_codes_num[3]]
    fig1, ax1 = plt.subplots()
    ax1.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
    ax1.axis('equal')
    plt.show()

