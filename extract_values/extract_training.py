
# Paths of the txt files from which we have to extract the values

base_path = '/home/saransh/custom_openfl_code/'

# We want to extract only the relevant information from the log files that were saved

# For that we will use a combination of bash scripting and python to extract the values out in a pandas dataframe

# We will first write a function that will use linux terminal to filter out all the irrelevant information

def get_relevant_information(seed, path=base_path):
    """Helper function to extract only the relevant information out from the log files that are generated

    Args:
        seed (int): The seed for which the experiment was run
        path (string, optional): The path to the base folder which contains all the subfolders. Defaults to base_path.
    """    
    import os
    os.system(f'cat {path}custom_code_seed_{seed}/agg_seed_{seed}_epochs_40.txt | grep "localy_tuned_model_validate task metrics..." -A1 | grep "dice_coef:" > {path}custom_code_seed_{seed}/agg_loc_tuned.txt ')
    os.system(f'cat {path}custom_code_seed_{seed}/agg_seed_{seed}_epochs_40.txt | grep "aggregated_model_validate task metrics..." -A1 | grep "dice_coef:" > {path}custom_code_seed_{seed}/agg_agg.txt')
    os.system(f'cat {path}custom_code_seed_{seed}/col_one_seed_{seed}_epochs_40.txt | grep "INFO     Sending metric for task aggregated_model_validate," > {path}custom_code_seed_{seed}/col_one_agg.txt')
    os.system(f'cat {path}custom_code_seed_{seed}/col_one_seed_{seed}_epochs_40.txt | grep "INFO     Sending metric for task localy_tuned_model_validate," > {path}custom_code_seed_{seed}/col_one_loc.txt')
    os.system(f'cat {path}custom_code_seed_{seed}/col_two_seed_{seed}_epochs_40.txt | grep "INFO     Sending metric for task aggregated_model_validate," > {path}custom_code_seed_{seed}/col_two_agg.txt')
    os.system(f'cat {path}custom_code_seed_{seed}/col_two_seed_{seed}_epochs_40.txt | grep "INFO     Sending metric for task localy_tuned_model_validate," > {path}custom_code_seed_{seed}/col_two_loc.txt')


# print('Starting the script')
# base_path = '/home/saransh/custom_openfl_code/'
# seed=49
# path = base_path + f'custom_code_seed_{seed}/' + 'col_one_agg.txt'
# lines = []
# with open(path,'r') as f:
#     l = f.readlines()
#     lines.append(l)

# curr = lines[0]
# relevant_lines = []
# for i in range(len(curr)):
#     if curr[i].find('dice_coef')!=-1:
#         relevant_lines.append(curr[i][curr[i].find('dice_coef'):].split()[1])

# print(relevant_lines)

def extract_agg_training(seed, base_path=base_path):
    import pandas as pd

    print('Extracting the col agg data...')
    col_one_agg_file = base_path + f'custom_code_seed_{seed}/' + 'col_one_agg.txt'
    col_two_agg_file = base_path + f'custom_code_seed_{seed}/' + 'col_two_agg.txt'
    temp_1 = []
    temp_2 = []
    with open(col_one_agg_file,'r') as f:
        l = f.readlines()
        temp_1.append(l)
    with open(col_two_agg_file,'r') as f:
        l = f.readlines()
        temp_2.append(l)
    temp_1 = temp_1[0]
    temp_2 = temp_2[0]
    col_one_agg = []
    col_two_agg = []
    for i in range(len(temp_1)):
        if temp_1[i].find('dice_coef')!=-1:
            col_one_agg.append(float(temp_1[i][temp_1[i].find('dice_coef'):].split()[1]))
    del temp_1
    for i in range(len(temp_2)):
        if temp_2[i].find('dice_coef')!=-1:
            col_two_agg.append(float(temp_2[i][temp_2[i].find('dice_coef'):].split()[1]))
    del temp_2

    print('Extracting col local data')
    col_one_loc_file = base_path + f'custom_code_seed_{seed}/' + 'col_one_loc.txt'
    col_two_loc_file = base_path + f'custom_code_seed_{seed}/' + 'col_two_loc.txt'
    temp_1 = []
    temp_2 = []
    with open(col_one_loc_file,'r') as f:
        l = f.readlines()
        temp_1.append(l)
    with open(col_two_loc_file,'r') as f:
        l = f.readlines()
        temp_2.append(l)
    temp_1 = temp_1[0]
    temp_2 = temp_2[0]
    col_one_loc = []
    col_two_loc = []
    for i in range(len(temp_1)):
        if temp_1[i].find('dice_coef')!=-1:
            col_one_loc.append(float(temp_1[i][temp_1[i].find('dice_coef'):].split()[1]))
    del temp_1
    for i in range(len(temp_2)):
        if temp_2[i].find('dice_coef')!=-1:
            col_two_loc.append(float(temp_2[i][temp_2[i].find('dice_coef'):].split()[1]))
    del temp_2

    print('Extracting agg data')
    agg_agg_file = base_path + f'custom_code_seed_{seed}/' + 'agg_agg.txt'
    agg_loc_file = base_path + f'custom_code_seed_{seed}/' + 'agg_loc_tuned.txt'
    temp_1 = []
    temp_2 = []
    with open(agg_agg_file,'r') as f:
        l = f.readlines()
        temp_1.append(l)
    with open(agg_loc_file,'r') as f:
        l = f.readlines()
        temp_2.append(l)
    temp_1 = temp_1[0]
    temp_2 = temp_2[0]
    agg_agg = []
    agg_loc = []
    for i in range(len(temp_1)):
        agg_agg.append(float(temp_1[i].split()[-1]))
    for i in range(len(temp_2)):
        agg_loc.append(float(temp_2[i].split()[-1]))

    epochs = [i for i in range(1,41)]
    df = pd.DataFrame()
    df['epochs'] = epochs
    df['col_one_agg'] = col_one_agg
    df['col_two_agg'] = col_two_agg
    df['col_one_loc'] = col_one_loc
    df['col_two_loc'] = col_two_loc
    df['agg_loc'] = agg_loc
    df['agg_agg'] = agg_agg
    # print(f'{df.head}')
    df.to_csv(f'{base_path}custom_code_seed_{seed}/final_results.csv',index=False)

# Calling the function to generate all the relevant files
# get_relevant_information(seed=2)
# get_relevant_information(seed=5)
# get_relevant_information(seed=9)
# get_relevant_information(seed=17)
# get_relevant_information(seed=37)
# get_relevant_information(seed=49)

# Making the aggregated results
# extract_agg_training(seed=2)
# extract_agg_training(seed=5)
# extract_agg_training(seed=9)
# extract_agg_training(seed=17)
# extract_agg_training(seed=37)
# extract_agg_training(seed=49)

if __name__ == '__main__':

    # print('Starting the script')
    # base_path = '/home/saransh/custom_openfl_code/'
    # seed=49
    # path = base_path + f'custom_code_seed_{seed}/' + 'col_one_agg.txt'
    # lines = []
    # with open(path,'r') as f:
    #     l = f.readlines()
    #     lines.append(l)

    # curr = lines[0]
    # relevant_lines = []
    # for i in range(len(curr)):
    #     if curr[i].find('dice_coef')!=-1:
    #         relevant_lines.append(curr[i][curr[i].find('dice_coef'):].split()[1])

    # print(relevant_lines)
    # extract_agg_training(seed=49)
    pass