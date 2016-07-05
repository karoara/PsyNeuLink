from Functions.System import System_Base
from Functions.Process import Process_Base
from Functions.Mechanisms.DDM import *
from Functions.Mechanisms.SigmoidLayer import *
from Functions.Mechanisms.AdaptiveIntegrator import *
from Functions.Mechanisms.EVCMechanism import *
from Globals.Keywords import *
from Functions.Utility import UtilityRegistry
from Functions.MechanismStates.MechanismState import MechanismStateRegistry

#region Preferences
DDM_prefs = FunctionPreferenceSet(
                prefs = {
                    kpVerbosePref: PreferenceEntry(True,PreferenceLevel.INSTANCE),
                    kpReportOutputPref: PreferenceEntry(True,PreferenceLevel.INSTANCE)})

process_prefs = FunctionPreferenceSet(reportOutput_pref=PreferenceEntry(True,PreferenceLevel.INSTANCE),
                                      verbose_pref=PreferenceEntry(True,PreferenceLevel.INSTANCE))
#endregion

#region Mechanisms
Input = SigmoidLayer(name='Input')
Reward = SigmoidLayer(name='Reward')
StimulusPrediction = AdaptiveIntegratorMechanism(name='StimPrediction')
RewardPrediction = AdaptiveIntegratorMechanism(name='RewardPrediction')
Decision = DDM(params={kwExecuteMethodParams:{kwDDM_DriftRate:(1.0, kwControlSignal),
                                                 kwDDM_Threshold:(10.0, kwControlSignal)},
                          kwDDM_AnalyticSolution:kwDDM_BogaczEtAl},
                  prefs = DDM_prefs,
                  name='Decision'
                  )
#endregion

#region Processes
TaskExecutionProcess = Process_Base(default_input_value=[0],
                                    params={kwConfiguration:[(Input, 0),
                                                             kwIdentityMatrix,
                                                             (Decision, 0)]}, # WILL THIS GET TWO inputStates IN EVC?
                                    prefs = process_prefs,
                                    name = 'TaskExecutionProcess')

RewardProcess = Process_Base(default_input_value=[0],
                             params={kwConfiguration:[(Reward, 1),
                                                      kwIdentityMatrix,
                                                      (RewardPrediction, 1)]},
                             prefs = process_prefs,
                             name = 'RewardProcess')

StimulusPredictionProcess = Process_Base(default_input_value=[0],
                                         params={kwConfiguration:[(Input, 0),
                                                                  kwIdentityMatrix,
                                                                  (StimulusPrediction, 2),
                                                                  kwIdentityMatrix,
                                                                  (Decision, 2)]}, # WILL THIS GET TWO inputStates IN EVC?
                                         prefs = process_prefs,
                                         name = 'StimulusPredictionProcess')
#endregion

#region System
mySystem = System_Base(params={kwProcesses:[TaskExecutionProcess, RewardProcess, StimulusPredictionProcess]},
                       name='EVC Test System')
#endregion

#region Inspect
mySystem.inspect()
mySystem.controller.inspect()
#endregion

#region Run
CentralClock.time_step = 0

# Present stimulus:
mySystem.execute([[1],[0],[0]])

# Present feedback:
CentralClock.time_step = 1
mySystem.execute([[0],[1],[0]])

# Run EVC:
CentralClock.time_step = 2
mySystem.execute([[0],[0],[0]])

#endregion
