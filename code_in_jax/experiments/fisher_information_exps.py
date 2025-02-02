#%%
import numpy             as np
import scipy.stats       as stats
import matplotlib.pyplot as plt
from   scipy.stats       import entropy


#%%


#%%
def calculate_kl_divergence(policy_output_p, policy_output_q):
    policy_output_p = np.array(policy_output_p)
    policy_output_q = np.array(policy_output_q)
    policy_output_q = np.where(policy_output_q==0, 1e-9, policy_output_q) # preventing division by zero
    kl_divergence   = entropy(policy_output_p, policy_output_q)
    return kl_divergence

#%%


#%%
policy_output_p = np.array([0.9, 2.1])
policy_output_q = np.array([1.3, 1.7])

kl_value = calculate_kl_divergence(policy_output_p, policy_output_q)

print(f"Policy output p      : {policy_output_p}")
print(f"Policy output q      : {policy_output_q}")
print(f"KL Divergence (P||Q) : {kl_value}")


#%%


#%%
def approximate_kl_divergence_fim(policy_parameter_diff, fisher_information_matrix):
    parameter_diff = np.array(policy_parameter_diff)
    fim = np.array(fisher_information_matrix)
    # KL divergence approximation : (1/2)*Δθ^T*F*Δθ
    approx_kl = 0.5*parameter_diff.T @ fim @ parameter_diff
    return approx_kl

#%%
policy_params_old  = np.array([0.9, 2.1])
policy_params_new  = np.array([1.3, 1.7])
policy_params_diff = policy_params_new-policy_params_old # Δθ

fisher_information_matrix = np.array(
    [[0.5, 0.1],
     [0.1, 0.8]]
    )

approx_kl_value = approximate_kl_divergence_fim(policy_params_diff, fisher_information_matrix)
print(f"Policy Parameter Difference (Δθ): {policy_params_diff}")
print(f"Fisher Information Matrix (FIM):\n{fisher_information_matrix}")
print(f"Approximated KL Divergence: {approx_kl_value}")

#%%


#%%

