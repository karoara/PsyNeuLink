from psyneulink.components.mechanisms.processing.transfermechanism import TransferMechanism
from psyneulink.components.mechanisms.processing.integratormechanism import IntegratorMechanism
from psyneulink.components.functions.function import Linear

class TestParameterStates:
    def test_inspect_function_params_slope(self):
        # A = IntegratorMechanism(function=Linear)
        B = TransferMechanism()
        A = TransferMechanism()
        print()
        print("A.function_object.slope --> ", A.function_object.slope)
        print("B.function_object.slope --> ", B.function_object.slope)
        # print("A.function_object._slope --> ", A.function_object._slope)
        # print("A.function_object.mod_slope --> ", A.function_object.mod_slope)
        # print("A.function_object.base_value_slope --> ", A.function_object.base_value_slope)
        print("- - - - - SETTING A.function_object.slope = 0.2 - - - - -")
        A.function_object.slope = 0.2
        print("A.function_object.slope --> ", A.function_object.slope)
        print("B.function_object.slope --> ", B.function_object.slope)
        # print("A.function_object._slope --> ", A.function_object._slope)
        # print("A.function_object.mod_slope --> ", A.function_object.mod_slope)

        print("- - - - - SETTING B.function_object.slope = 0.8 - - - - -")
        B.function_object.slope = 0.8
        print("A.function_object.slope --> ", A.function_object.slope)
        print("B.function_object.slope --> ", B.function_object.slope)
        # print("A.function_object.base_value_slope --> ", A.function_object.base_value_slope)
        print("- - - - - EXECUTING A - - - - -")
        print(A.execute(1.0))
        print("- - - - - EXECUTING B - - - - -")
        print(B.execute(1.0))
        print("A.function_object.slope --> ", A.function_object.slope)
        print("B.function_object.slope --> ", B.function_object.slope)
        # print("A.function_object._slope --> ", A.function_object._slope)
        # print("A.function_object.mod_slope --> ", A.function_object.mod_slope)
        # print("A.function_object.base_value_slope --> ", A.function_object.base_value_slope)
        print("- - - - - SETTING A.function_object.slope = 0.5 - - - - -")
        A.function_object.slope = 0.5
        print("A.function_object.slope --> ", A.function_object.slope)
        print("B.function_object.slope --> ", B.function_object.slope)

    def test_inspect_mechanism_params_noise(self):
        print("\n\n========================== starting second test ==========================\n\n")
        C = TransferMechanism(function=Linear(slope=2.0))
        # C = TransferMechanism()
        print("C.function_object.slope --> ", C.function_object.slope)
        print("executing: ", C.execute(1.0))
        C.function_object.slope=2.0
        print("setting slope to 2.0")
        print("executing: ", C.execute(1.0))
        print("C.noise --> ", C.noise)
        print("C._noise --> ", C._noise)
        # print("C.mod_noise --> ", C.mod_noise)
        # print("C.base_value_noise --> ", C.base_value_noise)
        print("- - - - - SETTING C.noise = 0.2 - - - - -")
        C.noise = 0.2
        print("C.user_params -->", C.user_params)
        print("C.noise --> ", C.noise)
        print("C._noise --> ", C._noise)
        # print("C.mod_noise --> ", C.mod_noise)
        # print("C.base_value_noise --> ", C.base_value_noise)
        print("- - - - - EXECUTING A - - - - -")
        C.execute(1.0)
        print("C.noise --> ", C.noise)
        print("C._noise --> ", C._noise)
        # print("C.mod_noise --> ", C.mod_noise)