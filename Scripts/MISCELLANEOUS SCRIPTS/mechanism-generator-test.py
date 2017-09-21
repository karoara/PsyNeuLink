from psyneulink.components.mechanisms.processing.transfermechanism import *
from psyneulink.components.process import Process
from psyneulink.components.system import System
from psyneulink.globals.keywords import *
from psyneulink.mechanismGenerator import mechanismGenerator

mechanism1 = TransferMechanism(name='my_Transfer1',
                       default_variable = [0],
                       function=Linear(),
                       time_constant =0.0,
                       params= {NOISE: 25.0}

                       )
# print(mechanism1.execute(100)," = value of executing mechanism1 independently with input of 100")

mechanism2 = TransferMechanism(name='my_Transfer2',
                               default_variable=[0],
                               function=Linear(slope=2.0),
                               time_constant=0.0

                               )

# print(mechanism2.execute(125)," = value of executing mechanism2 independently with input of 100")

mechanism3 = TransferMechanism(name='my_Transfer3',
                               default_variable=[0],
                               function=Linear(slope=2.0),
                               time_constant=0.0

                               )
# print(mechanism3.execute(250)," = value of executing mechanism3 independently with input of 200")

mechanism4 = TransferMechanism(name='my_Transfer4',
                               default_variable=[0],
                               function=Linear(slope=2.0),
                               time_constant=0.0

                               )
# print(mechanism4.execute(500)," = value of executing mechanism4 independently with input of 400")


mechanism5 = TransferMechanism(name='my_Transfer5',
                               default_variable=[0],
                               function=Linear(),
                               time_constant=0.0

                               )

# print(mechanism5.execute(1000)," = value of executing mechanism5 independently with input of 800")

mech1tuple = mechanism1, {PARAMETER_STATE_PARAMS:{FUNCTION_PARAMS:{SLOPE:10.0, INTERCEPT: 15.0}}}
mech2tuple = mechanism2, {PARAMETER_STATE_PARAMS:{FUNCTION_PARAMS:{SLOPE:50.0, INTERCEPT: 45.0}}}

path = [mech1tuple, mechanism2, mechanism3, mechanism4, mechanism5]
process1 = Process(default_variable=[100],
                 params={PATHWAY:path},
                 prefs={kpVerbosePref: PreferenceEntry(False, PreferenceLevel.INSTANCE),
                        kpReportOutputPref: PreferenceEntry(True, PreferenceLevel.INSTANCE)})
print(process1.runtime_params_dict)
print(process1.pathway)

system1 = System(processes=[process1], scheduler= mechanismGenerator(path))

print(system1.execute([[100]]), " = value of executing system with input of 100")