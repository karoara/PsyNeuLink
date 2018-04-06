import numpy as np
import psyneulink as pnl

# Control Parameters
#signalSearchRange = np.arange(1.0,2.1,0.5) # why 0.8 to 2.0 in increments of 0.2


#projection weights
letter_attention = 2.0
letter_output = 1.5

# Attention Mechanism
attention_layer = pnl.RecurrentTransferMechanism(name='Attention LAYER', function=pnl.Logistic(), hetero=-1.0)
attention_layer.set_log_conditions('value')

# Letter Mechanism
letter_layer = pnl.RecurrentTransferMechanism(name='Letter LAYER', function=pnl.Logistic(), hetero=-2.0)
letter_layer.set_log_conditions('value')




# Output Mechanisms
output_module = pnl.RecurrentTransferMechanism(name='Output Module',
                                   function=pnl.Logistic(), hetero=-3.0)
 #                                      slope=(1.0, pnl.ControlProjection(
#                                           control_signal_params={
#                                               pnl.ALLOCATION_SAMPLES: signalSearchRange}))))

output_module.set_log_conditions('value')
output_module.loggable_items

# Projection weights from attention to letter
letter_attention_weights = np.array([[letter_attention, letter_attention, 0.0, 0.0, 0.0, 0.0 ],
                           [0.0, 0.0,letter_attention, letter_attention, 0.0, 0.0],
                           [0.0, 0.0, 0.0, 0.0, letter_attention, letter_attention]])

# Projection weights from letter to output

letter_output_weights = np.array([[letter_output, 0.0],
                                    [0.0, letter_output],
                                    [letter_output, 0.0],
                                    [0.0, letter_output],
                                    [letter_output, 0.0],
                                    [0.0, letter_output]])

letter_output_process = pnl.Process(pathway=[letter_layer,
                                    output_module],
                           name='LETTER-OUTPUT PROCESS')
attention_letter_process = pnl.Process(pathway=[attention_layer,
                                                letter_layer],
                           name='LETTER-OUTPUT PROCESS')
# System:
mySystem = pnl.System(processes=[letter_output_process,
                                 attention_letter_process],
                      name='Eriksen System')

#    controller=pnl.EVCControlMechanism(prefs={pnl.LOG_PREF: pnl.PreferenceEntry(pnl.LogCondition.INITIALIZATION, pnl.PreferenceLevel.INSTANCE)}),
 #   enable_controller=True,
  #  monitor_for_control=[
   #     # (None, None, np.ones((1,1))),
    #    Reward,
     #   Decision.PROBABILITY_UPPER_THRESHOLD,
      #  ('OFFSET_RT', 1, -1),
#    ],
#    name='EVC Markus System')

# Show characteristics of system:
mySystem.show()

# Show graph of system
mySystem.show_graph()

nTrials = 2
letter_input = [0.15, 0.15, 0.15, 0.0, 0.15, 0.0]
attention_input = [0.03, 0.03, 0.03, 0.03]

stim_list_dict = {
    letter_layer: letter_input,
    attention_layer: attention_input}

mySystem.run(num_trials=nTrials,inputs=stim_list_dict)



#log input state of mySystem
#mySystem.controller.loggable_items
#mySystem.controller.set_log_conditions('InputState-0')
#mySystem.controller.set_log_conditions('value')

#mySystem.controller.set_log_conditions('Flanker Representation[slope] ControlSignal')
#mySystem.controller.set_log_conditions('Target Representation[slope] ControlSignal')

#mySystem.controller.objective_mechanism.set_log_conditions('value')
# print('current input value',mySystem.controller.input_states.values)
# print('current objective mech output value',mySystem.controller.objective_mechanism.output_states.values)
#


# configure EVC components
#mySystem.controller.control_signals[0].intensity_cost_function = pnl.Exponential(rate=0.8046).function
#mySystem.controller.control_signals[1].intensity_cost_function = pnl.Exponential(rate=0.8046).function
#
# #change prediction mechanism function_object.rate for all 3 prediction mechanisms
#
#mySystem.controller.prediction_mechanisms.mechanisms[0].function_object.rate = 1.0
#mySystem.controller.prediction_mechanisms.mechanisms[1].function_object.rate = 0.3481  # reward rate
#mySystem.controller.prediction_mechanisms.mechanisms[2].function_object.rate = 1.0





# log predictions:
#how to log this??? with prefs??
#mySystem.controller.prediction_mechanisms.mechanisms.


# add weight matrix for input updates here ! ??? ask Sebastian on march 9!

# W_new = W_hat_old + alpha*(W_hat_predicted - W_actual)




# for mech in mySystem.controller.prediction_mechanisms.mechanisms:
#     if mech.name == 'Flanker Stimulus Prediction Mechanism' or mech.name == 'Target Stimulus Prediction Mechanism':
#         # when you find a key mechanism (transfer mechanism) with the correct name, print its name
#         # print(mech.name)
#         mech.function_object.rate = 1.0
#
#     if 'Reward' in mech.name:
#         # print(mech.name)
#         mech.function_object.rate = 1.0
#         # mySystem.controller.prediction_mechanisms[mech].parameterStates['rate'].base_value = 1.0
#



#Markus: incongruent trial weights:

# f = np.array([1,1])
# # W_inc = np.array([[1.0, 0.0],[0.0, 1.5]])
# # W_con = np.array([[1.0, 0.0],[1.5, 0.0]])
#
#
# # generate stimulus environment: remember that we add one congruent stimulus infront of actuall stimulus list
# # compatible with MATLAB stimulus list for initialization
# nTrials = 4
# targetFeatures = [1.0, 1.0, 1.0, 1.0, 1.0,1.0, 1.0, 1.0, 1.0]
# flankerFeatures = np.array([1.0, 1.0, 1.0, 1.0, 1.0,1.0, 1.0, 1.0, 1.0])
# weights = np.array([1.0, 1.0, -1.0, 1.0, 1.0,-1.0, -1.0, 1.0, -1.0])
#
#
# flankerFeatures_inc = flankerFeatures *weights
# flankerFeatures_inc = np.ndarray.tolist(flankerFeatures_inc)
# # flankerFeatures_con = [1.5, 0]
# reward = [100, 100, 100, 100, 100, 100, 100, 100, 100]
#
#
# targetInputList = targetFeatures
# flankerInputList = flankerFeatures_inc
# rewardList = reward
#
# stim_list_dict = {
#     Target_Stim: targetInputList,
#     Flanker_Stim: flankerInputList,
#     Reward: rewardList
#
# }
# # mySystem.controller.objective_mechanism.loggable_items
# mySystem.run(num_trials=nTrials,inputs=stim_list_dict)

# Flanker_Rep.log.print_entries()
# Target_Rep.log.print_entries()
#Decision.log.print_entries()

# print('output state of objective mechanism', mySystem.controller.objective_mechanism.output_states.values)
#
# print('input state of EVC Control mechanism', mySystem.controller.input_state.value)
#
# print('mapping projection from objective mechanism to EVC Control mechanism',mySystem.controller.projections[0].matrix)

#mySystem.controller.log.print_entries()

#Reward.log.print_entries()
