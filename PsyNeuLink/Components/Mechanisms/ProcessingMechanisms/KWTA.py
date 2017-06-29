# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

# **************************************** KWTA *************************************************

from PsyNeuLink.Components.Functions.Function import Logistic
from PsyNeuLink.Globals.Keywords import *
from PsyNeuLink.Components.Mechanisms import Mechanism
from PsyNeuLink.Components.Mechanisms.ProcessingMechanisms.RecurrentTransferMechanism import *
from PsyNeuLink.Components.Projections.PathwayProjections.MappingProjection import MappingProjection
from PsyNeuLink.Components.States.InputState import InputState

class KWTAError(Exception):
    def __init__(self, error_value):
        self.error_value = error_value

    def __str__(self):
        return repr(self.error_value)

class KWTA(RecurrentTransferMechanism):

    componentType = KWTA

    paramClassDefaults = RecurrentTransferMechanism.paramClassDefaults.copy()

    standard_output_states = RecurrentTransferMechanism.standard_output_states.copy()

    @tc.typecheck
    def __init__(self,
                 default_input_value=None,
                 size=None,
                 input_states: tc.optional(tc.any(list, dict)) = None,
                 function=Logistic,
                 initial_value=None,
                 matrix=None,  # None defaults to a hollow uniform inhibition matrix
                 decay: tc.optional(tc.any(int, float)) = 1.0,
                 noise: is_numeric_or_none = 0.0,
                 time_constant: is_numeric_or_none = 1.0,
                 k_value: is_numeric_or_none = 0,
                 threshold: is_numeric_or_none = 0,
                 ratio: is_numeric_or_none = 0.5,
                 range=None,
                 output_states: tc.optional(tc.any(list, dict)) = [RESULT],
                 time_scale=TimeScale.TRIAL,
                 params=None,
                 name=None,
                 prefs: is_pref_set = None,
                 context=componentType + INITIALIZING,
                 ):

        # IMPLEMENTATION NOTE: parts of this region may be redundant with code in TransferMechanism.__init__()
        # region Fill in and infer default_input_value and size if they aren't specified in args
        if default_input_value is None and size is None:
            if matrix is None:
                default_input_value = [[0]]
                size = [1]
            else:
                try:
                    if isinstance(matrix, list):
                        size = len(matrix[0])
                    else:
                        size = matrix.shape[0]
                except:
                    raise KWTAError("Unable to parse matrix argument, {}. Please input a 2D array,"
                                    " list, or numpy matrix".format(matrix))

        # 6/23/17: This conversion is safe but likely redundant. If, at some point in development, size and
        # default_input_value are no longer 2D or 1D arrays, this conversion should still be safe, but wasteful.
        # region Convert default_input_value (if given) to a 2D array, and size (if given) to a 1D integer array

        try:
            if default_input_value is not None:
                default_input_value = np.atleast_2d(default_input_value)
                if len(np.shape(default_input_value)) > 2:  # number of dimensions of default_input_value > 2
                    warnings.warn("default_input_value had more than two dimensions (had {} dimensions) "
                                  "so only the first element of its second-highest-numbered axis will be"
                                  " used".format(len(np.shape(default_input_value))))
                    while len(np.shape(default_input_value)) > 2:  # reduce the dimensions of default_input_value
                        default_input_value = default_input_value[0]
        except:
            raise TransferError("Failed to convert default_input_value (of type {})"
                                " to a 2D array".format(type(default_input_value)))

        try:
            if size is not None:
                size = np.atleast_1d(size)
                if len(np.shape(size)) > 1:  # number of dimensions of size > 1
                    warnings.warn("size had more than one dimension (size had {} dimensions), so only the first "
                                  "element of its highest-numbered axis will be used".format(len(np.shape(size))))
                    while len(np.shape(size)) > 1:  # reduce the dimensions of size
                        size = size[0]
        except:
            raise TransferError("Failed to convert size (of type {}) to a 1D array.".format(type(size)))

        try:
            if size is not None:
                map(lambda x: int(x), size)  # convert all elements of size to int
        except:
            raise TransferError("Failed to convert an element in size to an integer.")
        # endregion

        # region If default_input_value is None, make it a 2D array of zeros each with length=size[i]
        # implementation note: for good coding practices, perhaps add setting to enable
        # easy change of default_input_value's default value, which is an array of zeros at the moment
        # added 6/22/17
        if default_input_value is None and size is not None:
            try:
                default_input_value = []
                for s in size:
                    default_input_value.append(np.zeros(s))
                default_input_value = np.array(default_input_value)
            except:
                raise TransferError("default_input_value was not specified, but PsyNeuLink was unable to "
                                    "infer default_input_value from the size argument, {}. size should be"
                                    " an integer or an array or list of integers. Either size or "
                                    "default_input_value must be specified.".format(size))
        # endregion

        # region If size is None, then make it a 1D array of scalars with size[i] = length(default_input_value[i])
        # added 6/22/17
        if size is None:
            size = []
            try:
                for input_vector in default_input_value:
                    size.append(len(input_vector))
                size = np.array(size)
            except:
                raise TransferError("size was not specified, but PsyNeuLink was unable to infer size from "
                                    "the default_input_value argument, {}. default_input_value can be an array,"
                                    " list, a 2D array, a list of arrays, array of lists, etc. Either size or"
                                    " default_input_value must be specified.".format(default_input_value))
        # endregion

        # region If length(size) = 1 and default_input_value is not None, then expand size to length(default_input_value)
        if len(size) == 1 and len(default_input_value) > 1:
            new_size = np.empty(len(default_input_value))
            new_size.fill(size[0])
            size = new_size
        # endregion

        # IMPLEMENTATION NOTE: if default_input_value and size are both specified as arguments, they will be checked
        # against each other in Component.py, during _instantiate_defaults().
        # endregion

        # region set up the additional input_state that will represent inhibition
        # convert input_states to a list, if it was a dict before: this makes working with it easier
        if isinstance(input_states, dict):
            input_states = [input_states]

        if input_states is None or len(input_states) == 0:
            input_states = ["Default_input_state"]

        if isinstance(input_states, list) and len(input_states) > 1:
            warnings.warn("kWTA adjusts only the FIRST input state. If you have multiple input states, "
                          "only the primary one will have adjusted, to have k values above the threshold.")

        input_states.append("Inhibition input state")
        # endregion

        params = self._assign_args_to_param_dicts(input_states=input_states,
                                                  k_value=k_value,
                                                  threshold=threshold,
                                                  ratio=ratio)

        if matrix is None:
            matrix = np.full((size[0], size[0]), -1) * get_matrix(HOLLOW_MATRIX, size[0], size[0])

        super().__init__(default_input_value = default_input_value,
                         size = size,
                         input_states = input_states,
                         function = function,
                         initial_value = initial_value,
                         decay = decay,
                         noise = noise,
                         matrix = matrix,
                         time_constant = time_constant,
                         range = range,
                         output_states = output_states,
                         time_scale = time_scale,
                         params = params,
                         name = name,
                         prefs = prefs,
                         context = context)

    def _instantiate_input_states(self, context=None):
        # this code is copied heavily from InputState.py, devel branch 6/26/17
        # the reason for this is to override the param-check that causes InputState to throw an exception
        # because the number of input_states is different from the length of the mechanism's "variable"
        owner = self

        # extendedSelfVariable = list(self.variable)
        # extendedSelfVariable.append(np.zeros(self.size[0]))
        # extendedSelfVariable = np.array(extendedSelfVariable)
        # print("length of extendedSelfVariable", len(extendedSelfVariable))
        # print(extendedSelfVariable)

        from PsyNeuLink.Components.States.State import _instantiate_state_list
        state_list = _instantiate_state_list(owner=owner,
                                             state_list=owner.input_states,
                                             state_type=InputState,
                                             state_param_identifier=INPUT_STATES,
                                             constraint_value=self.variable,
                                             constraint_value_name="function variable",
                                             context=context)

        # FIX: 5/23/17:  SHOULD APPEND THIS TO LIST OF EXISTING INPUT_STATES RATHER THAN JUST ASSIGN;
        #                THAT WAY CAN USE INCREMENTALLY IN COMPOSITION
        # if context and 'COMMAND_LINE' in context:
        #     if owner.input_states:
        #         owner.input_states.extend(state_list)
        #     else:
        #         owner.input_states = state_list
        # else:
        #     if owner._input_states:
        #         owner._input_states.extend(state_list)
        #     else:
        #         owner._input_states = state_list

        # FIX: This is a hack to avoid recursive calls to assign_params, in which output_states never gets assigned
        # FIX: Hack to prevent recursion in calls to setter and assign_params
        if context and 'COMMAND_LINE' in context:
            owner.input_states = state_list
        else:
            owner._input_states = state_list

        # Check that number of input_states and their variables are consistent with owner.variable,
        #    and adjust the latter if not
        for i in range(len(owner.input_states)):
            input_state = owner.input_states[i]
            try:
                variable_item_is_OK = iscompatible(self.variable[i], input_state.value)
                if not variable_item_is_OK:
                    break
            except IndexError:
                variable_item_is_OK = False
                break

        if not variable_item_is_OK:
            old_variable = owner.variable
            new_variable = []
            for state_name, state in owner.input_states:
                new_variable.append(state.value)
            owner.variable = np.array(new_variable)
            if owner.verbosePref:
                warnings.warn("Variable for {} ({}) has been adjusted "
                              "to match number and format of its input_states: ({})".
                              format(old_variable, append_type_to_name(owner), owner.variable))

    # adds indexOfInhibitionInputState to the attributes of KWTA
    def _instantiate_attributes_before_function(self, context=None):

        super()._instantiate_attributes_before_function(context=context)

        # this index is saved so the KWTA mechanism knows which input state represents inhibition
        self.indexOfInhibitionInputState = len(self.input_states) - 1

        int_k_value = int(self.k_value[0])
        # ^ this is hacky but necessary for now, since something is
        # incorrectly turning self.k_value into an array of floats
        n = self.size[0]
        if (self.k_value[0] > 0) and (self.k_value[0] < 1):
            k = int(round(int_k_value * n))
        elif (int_k_value < 0):
            k = n - int_k_value
        else:
            k = int_k_value

        self.int_k = k

    def _instantiate_attributes_after_function(self, context=None):

        super(RecurrentTransferMechanism, self)._instantiate_attributes_after_function(context=context)

        if isinstance(self.matrix, MappingProjection):
            self.recurrent_projection = self.matrix

        else:
            self.recurrent_projection = _instantiate_recurrent_projection(self, self.matrix, context=context)

        self._matrix = self.recurrent_projection.matrix

    # this function returns the KWTA-scaled current_input, which is scaled based on
    # self.k_value, self.threshold, and self.ratio
    def _kwta_scale(self, current_input):

        k = self.int_k

        inhibVector = self.input_states[self.indexOfInhibitionInputState].value  # inhibVector is the inhibition input
        inhibVector = np.array(inhibVector)  # may be redundant
        if (inhibVector == 0).all():
            print("inhib vector was all zeros, so converting to ones")
            inhibVector = np.ones(int(self.size[0]))
        if (inhibVector > 0).all():
            print("inhibVector was all positive, so it will be multiplied by negative one")
            inhibVector = -1 * inhibVector
        if (inhibVector == 0).any():
            raise KWTAError("inhibVector contained some, but not all, zeros: not currently supported")
        if (inhibVector > 0).any():
            raise KWTAError("inhibVector was not all positive or all negative: not currently supported")
        if len(inhibVector) != len(current_input):
            raise KWTAError("The inhibition vector is of a different length than the primary input vector.")

        if not isinstance(current_input, np.ndarray):
            warnings.warn("input was not a numpy array: this may cause unexpected KWTA behavior")

        sortedInput = sorted(current_input, reverse=True)  # sortedInput is the values of current_input, sorted

        # current_input[indices[i - 1]] is the i-th largest element of current_input
        indices = []
        for i in range(int(self.size[0])):
            indices.append(np.where(current_input == sortedInput[i])[0][0])
        indices = np.array(indices)

        # scales[i] is the scale on inhibition that would put the (i+1)-th largest
        # element in current_input at the threshold
        scales = np.zeros(int(self.size[0]))
        for i in range(int(self.size[0])):
            inhib = inhibVector[indices[i]]
            if inhib == 0:
                pass
            else:
                scales[i] = (self.threshold - current_input[indices[i]]) / inhib

        # ratio determines where between the two scales our final scale will lie
        # for most situations where the inhibition vector is negative, a lower ratio means more inhibition
        sk = sorted(scales, reverse=True)[k]
        skMinusOne = sorted(scales, reverse=True)[k - 1]
        final_scale = sorted(scales, reverse=True)[k] * self.ratio + sorted(scales, reverse=True)[k - 1] * (1 - self.ratio)

        return current_input + final_scale * inhibVector

    # this is the exact same as _execute in TransferMechanism, except that this _execute calls _kwta_scale()
    # and implements decay as self.previous_input *= self.decay
    def _execute(self,
                variable=None,
                runtime_params=None,
                clock=CentralClock,
                time_scale = TimeScale.TRIAL,
                context=None):

        """Execute TransferMechanism function and return transform of input

        Execute TransferMechanism function on input, and assign to output_values:
            - Activation value for all units
            - Mean of the activation values across units
            - Variance of the activation values across units
        Return:
            value of input transformed by TransferMechanism function in outputState[TransferOuput.RESULT].value
            mean of items in RESULT outputState[TransferOuput.MEAN].value
            variance of items in RESULT outputState[TransferOuput.VARIANCE].value

        Arguments:

        # CONFIRM:
        variable (float): set to self.value (= self.input_value)
        - params (dict):  runtime_params passed from Mechanism, used as one-time value for current execution:
            + NOISE (float)
            + TIME_CONSTANT (float)
            + RANGE ([float, float])
        - time_scale (TimeScale): specifies "temporal granularity" with which mechanism is executed
        - context (str)

        Returns the following values in self.value (2D np.array) and in
            the value of the corresponding outputState in the self.outputStates dict:
            - activation value (float)
            - mean activation value (float)
            - standard deviation of activation values (float)

        :param self:
        :param variable (float)
        :param params: (dict)
        :param time_scale: (TimeScale)
        :param context: (str)
        :rtype self.outputState.value: (number)
        """

        # NOTE: This was heavily based on 6/20/17 devel branch version of _execute from TransferMechanism.py
        # Thus, any errors in that version should be fixed in this version as well.

        # FIX: ??CALL check_args()??

        # FIX: IS THIS CORRECT?  SHOULD THIS BE SET TO INITIAL_VALUE
        # FIX:     WHICH SHOULD BE DEFAULTED TO 0.0??
        # Use self.variable to initialize state of input


        if INITIALIZING in context:
            self.previous_input = self.variable

        if self.decay is not None and self.decay != 1.0:
            self.previous_input *= self.decay

        # FIX: NEED TO GET THIS TO WORK WITH CALL TO METHOD:
        time_scale = self.time_scale

        #region ASSIGN PARAMETER VALUES

        time_constant = self.time_constant
        range = self.range
        noise = self.noise

        #endregion

        #region EXECUTE TransferMechanism FUNCTION ---------------------------------------------------------------------

        # FIX: NOT UPDATING self.previous_input CORRECTLY
        # FIX: SHOULD UPDATE PARAMS PASSED TO integrator_function WITH ANY RUNTIME PARAMS THAT ARE RELEVANT TO IT

        # Update according to time-scale of integration
        if time_scale is TimeScale.TIME_STEP:

            if not self.integrator_function:

                self.integrator_function = AdaptiveIntegrator(
                                            self.input_state.value,
                                            initializer = self.initial_value,
                                            noise = self.noise,
                                            rate = self.time_constant
                                            )

            current_input = self.integrator_function.execute(self.input_state.value,
                                                        # Should we handle runtime params?
                                                             # params={INITIALIZER: self.previous_input,
                                                             #         INTEGRATION_TYPE: ADAPTIVE,
                                                             #         NOISE: self.noise,
                                                             #         RATE: self.time_constant}
                                                             # context=context
                                                             # name=Integrator.componentName + '_for_' + self.name
                                                             )

        elif time_scale is TimeScale.TRIAL:
            if self.noise_function:
                if isinstance(noise, (list, np.ndarray)):
                    new_noise = []
                    for n in noise:
                        new_noise.append(n())
                    noise = new_noise
                elif isinstance(variable, (list, np.ndarray)):
                    new_noise = []
                    for v in variable[0]:
                        new_noise.append(noise())
                    noise = new_noise
                else:
                    noise = noise()

            current_input = self.input_state.value + noise
        else:
            raise MechanismError("time_scale not specified for KWTA")

        # this is the primary line that's different in KWTA compared to TransferMechanism
        # this scales the current_input properly
        current_input = self._kwta_scale(current_input)

        self.previous_input = current_input

        # Apply TransferMechanism function
        output_vector = self.function(variable=current_input, params=runtime_params)

        # # MODIFIED  OLD:
        # if list(range):
        # MODIFIED  NEW:
        if range is not None:
        # MODIFIED  END
            minCapIndices = np.where(output_vector < range[0])
            maxCapIndices = np.where(output_vector > range[1])
            output_vector[minCapIndices] = np.min(range)
            output_vector[maxCapIndices] = np.max(range)

        return output_vector
        #endregion

@tc.typecheck
def _instantiate_recurrent_projection(mech: Mechanism_Base,
                                      matrix: is_matrix = FULL_CONNECTIVITY_MATRIX,
                                      context=None):
    """Instantiate a MappingProjection from mech to itself

    """

    if isinstance(matrix, str):
        size = len(mech.variable[0])
        matrix = get_matrix(matrix, size, size)

    return MappingProjection(sender=mech,
                             receiver=mech.input_states[mech.indexOfInhibitionInputState],
                             matrix=matrix,
                             name=mech.name + ' recurrent projection')