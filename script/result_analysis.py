import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
import os
import sys

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
            self.motor_opt_cost_log = self.try_to_get_item_from_log_dict('motor_opt_cost_log')
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
    
    
    def add_if_not_none(self, a, b):
        if a is not None:
            if b is not None:
                return a + b
            else:
                return a
        else:
            return b

    def get_success_rate(self, log_csv_path=None, max_round_num=8):
        valid_num = 0
        success_num = 0
        failure_code_1_num = 0
        failure_code_2_num = 0
        failure_code_3_num = 0
        failure_code_4_num = 0

        failure_codes_round = np.zeros((max_round_num, 4)) # 4 failure codes: 1, 2, 3, 4
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

    dataset_result_folder = '/media/clarence/Clarence/anything2robot_data/result'
    # dataset_result_folder = "/media/clarence/Clarence/anything2robot_data/standford_dogs/result_2024_10_27_dog100_no_fea"
    dataset_result_analysis = DatasetResultAnalysis(dataset_result_folder)

    ######## Success rate and time consumption
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

