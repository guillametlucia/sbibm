import sbibm
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from sbibm.metrics import c2st
import time 
import torch
task = sbibm.get_task("repressilator")  # See sbibm.get_available_tasks() for all tasks
prior = task.get_prior()
simulator = task.get_simulator()
observation = task.get_observation(num_observation=1)  # 10 per task
budget = [1_000,10_000,100_000]


# Create an empty dictionary to store the lists
#make a dictionary c2st values
#make a dictionary for each algorithm (keys)
#make  a list for each simulation budget (keys)with 10 values each for the 10 oibservations
print('a')
c2st_values = {} #make dictionary c2st vaalues
posteriors = {}
#make a dictionary for each algorithm (keys)
#list of algorithms
#algorithms = [rej_abc, smc, nle,snle,npe,snpe,nre,snre]

#import agorithms

#from sbibm.algorithms.sbi.npe import run as npe

#sequential
from sbibm.algorithms import snpe



algorithms = [snpe] 
names = ['npe']
params = ["n_tetR", "n_lacI", "n_lamdacI", "alpha_tetR", 
                "alpha_lacI","alpha_lambdacI","alpha_0_tetR",
                "alpha_0_lacI","alpha_0_lamdacI","beta_tetR",
                "beta_lacI","beta_lamdacI"]
times= {}

for i, algorithm in enumerate(algorithms): # Iterate over algorithms
    algo_dict= {}
    algorithm_name = names[i]  # Get the name of the algorithm function
    #iterate over budgets
    for b in budget:
        budget_dict={}
        #iterate over observations
        val_list =  []
        execution_times = [] # List to store execution times
        for i in range(4):
            start_time = time.process_time() # Start timing
            posterior_samples, _, _ = algorithm(task=task, num_samples=10_000, num_observation=i+1, num_simulations=b, num_rounds=1) #run algorithm and get posterior samples
            end_time = time.process_time() # End timing
            execution_time = end_time - start_time # Calculate elapsed time
            execution_times.append(execution_time) # Store execution time`
            
            posterior_samples_array = posterior_samples.numpy() #create numpy array of posterior torch tensor
            #add column to know where its coming from later 
            new_column = np.full((posterior_samples_array.shape[0], 1), 'predicted', dtype=str)
            # Concatenate the original array and the new column array
            df_post = pd.DataFrame(posterior_samples_array, columns=params[:12])
            
            df_post['Group'] = new_column
            post_ready = np.hstack((posterior_samples_array, new_column)) #horizontal stack

            reference_samples = task.get_reference_posterior_samples(num_observation=i+1) #get ref
            ref_array = reference_samples.numpy() #same as with posterior
            new_c = np.full((ref_array.shape[0],1),'reference',dtype=str)
            df_ref = pd.DataFrame(ref_array,columns=params[:12])
            df_ref['Group']= new_c

            df_post = pd.read_csv(f'df_posterior_{i}_1000_npe.csv')
            #concatenate to join both dataframes
            df_joint = pd.concat([df_post, df_ref], ignore_index=True, sort=False) #ignores index,not meaninful
            #add budget number column
            b_column = np.full((len(df_joint), 1), b)
            df_joint['Budget'] = b_column
            #get a sepaarate dataframe for each algorithm, observation and each budget
            #save as csv 
            df_joint.to_csv(f'df_posterior_{i + 1}_{b}_{algorithm_name}.csv', index=False)
            df_joint = pd.read_csv(f'df_posterior_{i + 1}_{b}_{algorithm_name}.csv')
            df_post = df_joint[df_joint['Group'] == 'p']
            posterior_samples = torch.tensor(df_post[params].values)
            #calculate accuracy 
            c2st_accuracy = c2st(reference_samples, posterior_samples) 
            val_list.append(float(c2st_accuracy)) #append accuracy values
        budget_dict['c2st_values'] = val_list #asign value list to key of budget 
        budget_dict['execution_times'] = execution_times # Store execution times in dictionary
        algo_dict[f'{b}'] = budget_dict
    c2st_values[algorithm_name] = algo_dict

       
# Create empty DataFrame
c2st_stats = pd.DataFrame()

# Iterate over algorithms
for algorithm, simulations in c2st_values.items():
    # Iterate over simulation budgets
    for num_simulations, budget_dict in simulations.items():
        c2st_values = budget_dict['c2st_values']
       # execution_times = budget_dict['execution_times']
        
        # Calculate mean and standard deviation of c2st values
        mean_c2st = np.mean(c2st_values)
        std_c2st = np.std(c2st_values)
        
        # Calculate mean execution time
        mean_execution_time = np.mean(execution_times)
        std_time = np.std(execution_times)
        for v in c2st_values:
            # Create a new row with algorithm, simulation budget, mean c2st, standard deviation c2st, and mean execution time
            row = {'Algorithm': algorithm, 'Simulation Budget': num_simulations, 'C2ST': mean_c2st, 'Standard Deviation C2ST': std_c2st, 'Mean Execution Time': mean_execution_time}
            # Append the row to the DataFrame
            c2st_stats = c2st_stats.append(row, ignore_index=True)


# Save DataFrame as CSV file
c2st_stats.to_csv('c2st_stats_onebyone.csv', index=False)
