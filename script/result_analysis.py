import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
import os
import sys
from progress.bar import IncrementalBar
import pandas as pd
file_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(file_path, '../auto_design'))
sys.path.append(os.path.join(file_path, '../auto_design/modules'))
import seaborn as sns

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
    


class FeaResultAnalysis:
    def __init__(self, fea_result_csv_path, key_words_stl_file=None):
        self.fea_result_csv_path = fea_result_csv_path
        '''
        csv file format:
        robot_name	round_id	stl_file	success_flag	best_relative_density
        '''
        self.fea_result_df = pd.read_csv(fea_result_csv_path)

        # Remove the rows where stl file is RR_UP.stl or RR_LOW.stl because they are not always 1 due to some bug in the FEA
        self.fea_result_df = self.fea_result_df[~self.fea_result_df['stl_file'].isin(['RR_UP.stl', 'RR_LOW.stl'])]


    def get_best_relative_density_array_of_key_words(self, key_words, key_word_column_name='stl_file'):
        # Get the best relative density array of each key word
        best_relative_density_array_of_key_words = {}
        for key_word in key_words:
            best_relative_density_array_of_key_words[key_word] = self.fea_result_df[self.fea_result_df[key_word_column_name] == key_word]['best_relative_density'].values

        return best_relative_density_array_of_key_words 


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
    fea_result_csv_path = os.path.join(dataset_result_folder, '../fea_result_2024Dec/fea_result.csv')
    
    max_round_num = 8
    
    # === Global Font Style Configuration ===
    title_fontsize = 8        # Title font size
    label_fontsize = 8        # Axis label font size
    tick_fontsize = 6         # Tick label size
    legend_fontsize = 8       # Legend font size
    suptitle_fontsize = 8     # Super title (figure-level title)
    font_family = 'DejaVu Sans'  # You can change this to 'Arial', 'Times New Roman', etc.

    line_width = 1.5
    marker_size = 4     

    # Apply font settings globally
    plt.rcParams.update({
        'font.family': font_family,
        'axes.titlesize': title_fontsize,
        'axes.labelsize': label_fontsize,
        'xtick.labelsize': tick_fontsize,
        'ytick.labelsize': tick_fontsize,
        'legend.fontsize': legend_fontsize,
        'figure.titlesize': suptitle_fontsize
    })

    
    # For bar plots and line plots (categorical data)
    # colors = plt.get_cmap('tab10').colors
    # colors = plt.cm.Set2(np.linspace(0, 1, 8))  # or Set3
    colors = sns.color_palette("muted", 8) 


    ########## FIGURE 1: 2x2 ##########
    fig1, axs1 = plt.subplots(2, 2, figsize=(7.8, 5), constrained_layout=True)

    #### Subplot 1: Motor Cost ####
    ax = axs1[0, 1]
    motor_position_cost_rounds_map, motor_occupancy_cost_rounds_map, _ = dataset_result_analysis.get_motor_cost(max_round_num=max_round_num)
    motor_position_cost_avg = np.zeros(max_round_num)
    motor_occupancy_cost_avg = np.zeros(max_round_num)
    motor_position_cost_std = np.zeros(max_round_num)
    motor_occupancy_cost_std = np.zeros(max_round_num)
    for i in range(1, max_round_num + 1):
        motor_position_cost_avg[i-1] = np.mean(motor_position_cost_rounds_map[i])
        motor_position_cost_std[i-1] = np.std(motor_position_cost_rounds_map[i])
        motor_occupancy_cost_avg[i-1] = np.mean(motor_occupancy_cost_rounds_map[i])
        motor_occupancy_cost_std[i-1] = np.std(motor_occupancy_cost_rounds_map[i])

    # Remove round 8 since it has no valid result
    motor_position_cost_avg = motor_position_cost_avg[:-1]
    motor_position_cost_std = motor_position_cost_std[:-1]
    motor_occupancy_cost_avg = motor_occupancy_cost_avg[:-1]
    motor_occupancy_cost_std = motor_occupancy_cost_std[:-1]

    x = np.arange(1, max_round_num)
    width = 0.35
    ax.bar(x - width/2, motor_position_cost_avg, width, yerr=motor_position_cost_std, capsize=5,
        color=colors[0], label="Position Cost")
    ax.bar(x + width/2, motor_occupancy_cost_avg, width, yerr=motor_occupancy_cost_std, capsize=5,
        color=colors[1], label="Occupancy Cost")
    ax.set_xlabel("Round")
    ax.set_ylabel("Cost")
    ax.set_title("Motor Costs per Round")
    ax.set_xticks(x)
    ax.set_ylim(0, 5)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend()

    #### Subplot 2: Mesh Similarity ####
    ax = axs1[1, 0]
    hausdorff_path = os.path.join(dataset_result_folder, 'hausdorff_distance_rounds_map.pkl')
    avg_point_path = os.path.join(dataset_result_folder, 'average_point_distance_rounds_map.pkl')
    if os.path.exists(hausdorff_path) and os.path.exists(avg_point_path):
        with open(hausdorff_path, 'rb') as f:
            hausdorff_map = pkl.load(f)
        with open(avg_point_path, 'rb') as f:
            avg_point_map = pkl.load(f)
    else:
        hausdorff_map, avg_point_map = dataset_result_analysis.get_mesh_similarity(max_round_num=max_round_num)
        with open(hausdorff_path, 'wb') as f:
            pkl.dump(hausdorff_map, f)
        with open(avg_point_path, 'wb') as f:
            pkl.dump(avg_point_map, f)

    avg_point_avg = np.zeros(max_round_num - 1)
    avg_point_std = np.zeros(max_round_num - 1)
    scaled_avg_point = np.zeros(max_round_num - 1)
    scale_factor = 1 / 1.1
    for i in range(1, max_round_num):
        data = np.array(avg_point_map[i])
        avg_point_avg[i-1] = data.mean()
        avg_point_std[i-1] = data.std()
        scaled_avg_point[i-1] = data.mean() * (scale_factor ** (i - 1))
    wider_width = 0.6
    ax.bar(x, avg_point_avg, wider_width, yerr=avg_point_std, capsize=5,
        color=colors[2], label="Avg Point Dist")
    ax.plot(x, scaled_avg_point, color=colors[3], marker="o", label="Scaled Avg Point Dist", linewidth=line_width, markersize=marker_size)
    ax.set_xlabel("Round")
    ax.set_ylabel("Distance")
    ax.set_title("Mesh Similarity per Round")
    ax.set_xticks(x)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend()

    #### Subplot 3: Success Rate + Time ####
    ax1_main = axs1[0, 0]
    ax2_twin = ax1_main.twinx()
    _, _, failure_codes_num, _, growing_success_rate = dataset_result_analysis.get_success_rate(
        log_csv_path=os.path.join(dataset_result_folder, 'result_log.csv'))
    avg_time, max_time, min_time, time_parts = dataset_result_analysis.get_time_consumption()
    # Convert time to minutes
    avg_time = avg_time / 60
    max_time = max_time / 60
    min_time = min_time / 60

    growing_avg = np.cumsum(avg_time)
    growing_max = np.cumsum(max_time)
    growing_min = np.cumsum(min_time)
    rounds = np.arange(max_round_num)
    bar_width = 0.6
    ax1_main.bar(rounds, growing_success_rate, bar_width, color=colors[0], alpha=0.5)
    ax2_twin.plot(rounds, growing_avg, color=colors[1], marker='o', label="Avg Time", linewidth=line_width, markersize=marker_size)
    ax2_twin.plot(rounds, growing_max, color=colors[2], marker='s', label="Max Time", linewidth=line_width, markersize=marker_size)
    ax2_twin.plot(rounds, growing_min, color=colors[3], marker='^', label="Min Time", linewidth=line_width, markersize=marker_size)
    ax1_main.set_xlabel("Round")
    ax1_main.set_xticklabels([str(i) for i in range(0, max_round_num + 1)])

    ax1_main.set_ylabel("Success Rate", color=colors[0])
    ax2_twin.set_ylabel("Time (min)", color=colors[1])
    ax1_main.set_title("Success Rate and Time per Round")
    ax2_twin.legend(loc="upper right")
    ax1_main.grid(True, linestyle='--', alpha=0.5)

    # axs1[1, 1] left blank intentionally

    #### Subplot 4: FEA Density presented by rounds ####
    ax = axs1[1, 1]
    fea_result_analysis = FeaResultAnalysis(fea_result_csv_path)
    key_words = [1, 2, 3, 4, 5, 6, 7]
    key_word_column_name = 'round_id'
    best_relative_density_array_of_key_words = fea_result_analysis.get_best_relative_density_array_of_key_words(key_words, key_word_column_name)
    print(best_relative_density_array_of_key_words)

    avg_best_relative_density_of_key_words = {}
    std_best_relative_density_of_key_words = {}
    for k in key_words:
        avg_best_relative_density_of_key_words[k] = np.mean(best_relative_density_array_of_key_words[k])
        std_best_relative_density_of_key_words[k] = np.std(best_relative_density_array_of_key_words[k])

    avg_list = [avg_best_relative_density_of_key_words[k] for k in key_words]
    std_list = [std_best_relative_density_of_key_words[k] for k in key_words]
    bar_x = np.arange(len(key_words))
    bar_width = 0.6
    ax.bar(bar_x, avg_list, bar_width, yerr=std_list, capsize=5, color=colors[4])
    ax.set_xticks(bar_x)
    ax.set_xticklabels(key_words)
    ax.set_ylim(0, 0.3)
    ax.set_ylabel("Optimized Relative Density")
    ax.set_xlabel("Round")
    ax.set_title("Optimized Relative Density by Round")
    ax.grid(True, linestyle='--', alpha=0.6)


    fig1.suptitle("Robot Design Metrics – Part 1", fontsize=suptitle_fontsize)
    fig1.savefig(os.path.join(dataset_result_folder, "robot_metrics_part1.png"), dpi=300)
    fig1.savefig(os.path.join(dataset_result_folder, "robot_metrics_part1.pdf"), dpi=300)

    ########## FIGURE 2: 1x3 ##########
    fig2, axs2 = plt.subplots(1, 3, figsize=(7.8, 2.5), constrained_layout=True)

    #### Subplot 1: Time per Part (Pie) ####
    ax = axs2[0]
    labels = ['Mesh Dec.', 'Motor Opt.', 'Key Voxel Srch', 'Intf Rem.', 'Density Opt.']
    avg_time_parts = np.mean(time_parts, axis=0)
    #ax.pie(avg_time_parts, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors, textprops={'fontsize': label_fontsize})
    # Adjust pie size and appearance
    ax.pie(avg_time_parts, 
           labels=labels, 
           autopct='%1.1f%%', 
           startangle=90, 
           colors=colors,
           radius=3,  # Adjust size of pie (0.8 = 80% of available space)
           textprops={'fontsize': label_fontsize},  # Control label text size
           pctdistance=0.75,  # Distance of percentage labels from center
           labeldistance=1.1)  # Distance of category labels from center
    
    # Optional: ensure pie is drawn as circle even with rectangular subplot
    ax.axis('equal')
    
    # Set font sizes
    ax.set_title("Average Time per Step", fontsize=title_fontsize)

    #### Subplot 2: Failure Reason Pie ####
    ax = axs2[1]
    failure_labels = ['Motor Cost Too High', 'Mesh Destroyed', 'FEA Not Feasible']
    failure_sizes = [failure_codes_num[0], failure_codes_num[1] + failure_codes_num[2], failure_codes_num[3]]
    #ax.pie(failure_sizes, labels=failure_labels, autopct='%1.1f%%', startangle=90, colors=colors, textprops={'fontsize': label_fontsize})
    
    # Adjust pie size and appearance
    ax.pie(failure_sizes, 
           labels=failure_labels, 
           autopct='%1.1f%%', 
           startangle=90, 
           colors=colors,
           radius=3,  # Adjust size of pie (0.8 = 80% of available space)
           textprops={'fontsize': label_fontsize},  # Control label text size
           pctdistance=0.75,  # Distance of percentage labels from center
           labeldistance=1.1)  # Distance of category labels from center
    
    # Optional: ensure pie is drawn as circle even with rectangular subplot
    ax.axis('equal')
    ax.set_title("Failure Reasons", fontsize=title_fontsize)

    #### Subplot 3: FEA Density Bar ####
    ax = axs2[2]
    key_words_stl_file = ['FR_LOW.stl', 'FL_LOW.stl', 'RL_LOW.stl', 'FL_UP.stl', 'FR_UP.stl', 'RL_UP.stl', 'BODY.stl']
    key_words = ['LOW', 'UP', 'BODY']
    fea_result_analysis = FeaResultAnalysis(fea_result_csv_path, key_words_stl_file)
    merged = {k: [] for k in key_words}
    best_density_map = fea_result_analysis.get_best_relative_density_array_of_key_words(key_words_stl_file)
    for k in key_words:
        for f in key_words_stl_file:
            if k in f:
                merged[k] = np.concatenate([merged[k], best_density_map[f]])
    avg_density = [np.mean(merged[k]) for k in key_words]
    std_density = [np.std(merged[k]) for k in key_words]
    bar_x = np.arange(len(key_words))
    bar_width = 0.6
    ax.bar(bar_x, avg_density, bar_width, yerr=std_density, capsize=5, color=colors[4])
    ax.set_xticks(bar_x)
    ax.set_xticklabels(key_words)
    ax.set_ylim(0, 0.3)
    ax.set_ylabel("Best Relative Density")
    ax.set_xlabel("Body Parts")
    ax.set_title("Best Relative Density by Part")
    ax.grid(True, linestyle='--', alpha=0.6)


    fig2.suptitle("Robot Design Metrics – Part 2", fontsize=suptitle_fontsize)
    fig2.savefig(os.path.join(dataset_result_folder, "robot_metrics_part2.png"), dpi=300)
    fig2.savefig(os.path.join(dataset_result_folder, "robot_metrics_part2.pdf"), dpi=300)

    plt.show()

