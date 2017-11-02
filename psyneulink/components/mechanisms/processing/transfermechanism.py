# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

# NOTES:
#  * COULD NOT IMPLEMENT integrator_function in paramClassDefaults (see notes below)
#  * NOW THAT NOISE AND TIME_CONSTANT ARE PROPRETIES THAT DIRECTLY REFERERNCE integrator_function,
#      SHOULD THEY NOW BE VALIDATED ONLY THERE (AND NOT IN TransferMechanism)??
#  * ARE THOSE THE ONLY TWO integrator PARAMS THAT SHOULD BE PROPERTIES??

# ********************************************  TransferMechanism ******************************************************

"""
..
    Sections:
      * :ref:`Transfer_Overview`
      * :ref:`Transfer_Creation`
      * :ref:`Transfer_Execution`
      * :ref:`Transfer_Class_Reference`

.. _Transfer_Overview:

Overview
--------

A TransferMechanism transforms its input using a simple mathematical function.  The input can be a single scalar value
or an an array of scalars (list or 1d np.array).  The function used to carry out the transformation can be selected
from a standard set of `Functions <Function>` (`Linear`, `Exponential` or `Logistic`) or specified using a
user-defined custom function.  The transformation can be carried out instantaneously or in a time-averaged manner,
as described in `Transfer_Execution`.

.. _Transfer_Creation:

Creating a TransferMechanism
-----------------------------

A TransferMechanism can be created directly by calling its constructor, or using the `mechanism` command and specifying
*TRANSFER_MECHANISM* as its **mech_spec** argument.  Its `function <TransferMechanism.function>` is specified in the
**function** argument, which can be the name of a `Function <Function>` class (first example below), or a call to its
constructor which can include arguments specifying the function's parameters (second example)::

    my_linear_transfer_mechanism = TransferMechanism(function=Linear)
    my_logistic_transfer_mechanism = TransferMechanism(function=Logistic(gain=1.0, bias=-4)

In addition to function-specific parameters, `noise <TransferMechanism.noise>` and `time_constant
<TransferMechanism.time_constant>` parameters can be specified for the Mechanism (see `Transfer_Execution`).


.. _Transfer_Structure:

Structure
---------

A TransferMechanism has a single `InputState`, the `value <InputState.InputState.value>` of which is
used as the `variable <TransferMechanism.variable>` for its `function <TransferMechanism.function>`. The
`function <TransferMechanism.function>` can be selected from one of three standard PsyNeuLink `Functions <Function>`:
`Linear`, `Logistic` or `Exponential`; or a custom function can be specified, so long as it returns a numeric value or
list or np.ndarray of numeric values.  The result of the `function <TransferMechanism.function>` is assigned as the
only item of the TransferMechanism's `value <TransferMechanism.value>` and as the `value <OutputState.value>` of its
`primary OutputState <OutputState_Primary>` (see `below <Transfer_OutputState>`).  Additional OutputStates can be
assigned using the TransferMechanism's `standard OutputStates <TransferMechanism_Standard_OutputStates>`
(see `OutputState_Standard`) or by creating `custom OutputStates <OutputState_Customization>`.

.. _Transfer_Execution:

Execution
---------

COMMENT:
DESCRIBE AS TWO MODES (AKIN TO DDM):  INSTANTANEOUS AND TIME-AVERAGED
INSTANTANEOUS:
input transformed in a single `execution <Transfer_Execution>` of the Mechanism)
TIME-AVERAGED:
input transformed using `step-wise` integration, in which each execution returns the result of a subsequent step of the
integration process).
COMMENT

When a TransferMechanism is executed, it transforms its input using its `function <TransferMechanism.function>` and
the following parameters (in addition to any specified for the `function <TransferMechanism.function>`):

    * `noise <TransferMechanism.noise>`: applied element-wise to the input before transforming it.
    ..
    * `range <TransferMechanism.range>`: caps all elements of the `function <TransferMechanism.function>` result by
      the lower and upper values specified by range.
    ..
    * `integrator_mode <TransferMechanism.integrator_mode>`: determines whether the input will be time-averaged before
      passing through the function of the mechanisms. When `integrator_mode <TransferMechanism.integrator_mode>` is set
      to True, the TransferMechanism exponentially time-averages its input before transforming it.
    ..
    * `time_constant <TransferMechanism.time_constant>`: if the `integrator_mode <TransferMechanism.integrator_mode>`
      attribute is set to True, the `time_constant <TransferMechanism.time_constant>` attribute is the rate of
      integration (a higher value specifies a faster rate); if `integrator_mode <TransferMechanism.integrator_mode>` is
      False, `time_constant <TransferMechanism.time_constant>` is ignored and time-averaging does not occur.



.. _Transfer_OutputState:

After each execution of the Mechanism the result of `function <TransferMechanism.function>` is assigned as the
only item of the Mechanism's `value <TransferMechanism.value>`, the `value <OutputState.value>` of its
`primary OutputState <OutputState_Primary>`, (same as the output_states[RESULT] OutputState if it has been assigned),
and to the 1st item of the Mechanism's `output_values <TransferMechanism.output_values>` attribute;

.. _Transfer_Class_Reference:

Class Reference
---------------

"""
import inspect
import numbers

import numpy as np
import typecheck as tc

from psyneulink.components.component import Component, function_type, method_type
from psyneulink.components.functions.function import AdaptiveIntegrator, Linear
from psyneulink.components.mechanisms.mechanism import MechanismError, Mechanism
from psyneulink.components.mechanisms.processing.processingmechanism import ProcessingMechanism_Base
from psyneulink.components.states.inputstate import InputState
from psyneulink.components.states.outputstate import \
    OutputState, PRIMARY_OUTPUT_STATE, StandardOutputStates, standard_output_states
from psyneulink.globals.keywords import FUNCTION, INITIALIZER, INITIALIZING, MEAN, MEDIAN, NOISE, RATE, RESULT, STANDARD_DEVIATION, TRANSFER_FUNCTION_TYPE, TRANSFER_MECHANISM, VARIANCE, kwPreferenceSetName
from psyneulink.globals.preferences.componentpreferenceset import is_pref_set, kpReportOutputPref, kpRuntimeParamStickyAssignmentPref
from psyneulink.globals.preferences.preferenceset import PreferenceEntry, PreferenceLevel
from psyneulink.globals.utilities import append_type_to_name, iscompatible
from psyneulink.scheduling.timescale import CentralClock, TimeScale

__all__ = [
    'INITIAL_VALUE', 'RANGE', 'TIME_CONSTANT', 'Transfer_DEFAULT_BIAS', 'Transfer_DEFAULT_GAIN', 'Transfer_DEFAULT_LENGTH',
    'Transfer_DEFAULT_OFFSET', 'TRANSFER_OUTPUT', 'TransferError', 'TransferMechanism',
]

import functools
import ctypes
import psyneulink.llvm as pnlvm
from llvmlite import ir

# TransferMechanism parameter keywords:
RANGE = "range"
TIME_CONSTANT = "time_constant"
INITIAL_VALUE = 'initial_value'

# TransferMechanism default parameter values:
Transfer_DEFAULT_LENGTH = 1
Transfer_DEFAULT_GAIN = 1
Transfer_DEFAULT_BIAS = 0
Transfer_DEFAULT_OFFSET = 0
# Transfer_DEFAULT_RANGE = np.array([])

# This is a convenience class that provides list of standard_output_state names in IDE
class TRANSFER_OUTPUT():
    """
    .. _TransferMechanism_Standard_OutputStates:

    `Standard OutputStates <OutputState_Standard>` for `TransferMechanism`: \n

    .. _TRANSFER_MECHANISM_RESULT:

    *RESULT* : 1d np.array
      result of `function <TransferMechanism.function>` (same as `value <TransferMechanism.value>`).

    .. _TRANSFER_MECHANISM_MEAN:

    *MEAN* : float
      mean of `value <TransferMechanism.value>`.

    .. _TRANSFER_MECHANISM_MEDIAN:

    *MEDIAN* : float
      median of `value <TransferMechanism.value>`.

    .. _TRANSFER_MECHANISM_STD_DEV:

    *STANDARD_DEVIATION* : float
      standard deviation of `value <TransferMechanism.value>`.

    .. _TRANSFER_MECHANISM_VARIANCE:

    *VARIANCE* : float
      variance of `output_state.value`.

    """
    RESULT=RESULT
    MEAN=MEAN
    MEDIAN=MEDIAN
    STANDARD_DEVIATION=STANDARD_DEVIATION
    VARIANCE=VARIANCE
# THE FOLLOWING WOULD HAVE BEEN NICE, BUT IDE DOESN'T EXECUTE IT, SO NAMES DON'T SHOW UP
# for item in [item[NAME] for item in DDM_standard_output_states]:
#     setattr(DDM_OUTPUT.__class__, item, item)


class TransferError(Exception):
    def __init__(self, error_value):
        self.error_value = error_value

    def __str__(self):
        return repr(self.error_value)

# IMPLEMENTATION NOTE:  IMPLEMENTS OFFSET PARAM BUT IT IS NOT CURRENTLY BEING USED
class TransferMechanism(ProcessingMechanism_Base):
    """
    TransferMechanism(           \
    default_variable=None,    \
    size=None,                   \
    function=Linear,             \
    initial_value=None,          \
    noise=0.0,                   \
    time_constant=1.0,           \
    integrator_mode=False,       \
    range=(float:min, float:max),\
    params=None,                 \
    name=None,                   \
    prefs=None)

    Subclass of `ProcessingMechanism <ProcessingMechanism>` that performs a simple transform of its input.

    COMMENT:
        Description
        -----------
            TransferMechanism is a Subtype of the ProcessingMechanism Type of the Mechanism Category of the
                Component class
            It implements a Mechanism that transforms its input variable based on FUNCTION (default: Linear)

        Class attributes
        ----------------
            + componentType (str): TransferMechanism
            + classPreference (PreferenceSet): Transfer_PreferenceSet, instantiated in __init__()
            + classPreferenceLevel (PreferenceLevel): PreferenceLevel.SUBTYPE
            + ClassDefaults.variable (value):  Transfer_DEFAULT_BIAS

        Class methods
        -------------
            None

        MechanismRegistry
        -----------------
            All instances of TransferMechanism are registered in MechanismRegistry, which maintains an
              entry for the subclass, a count for all instances of it, and a dictionary of those instances
    COMMENT

    Arguments
    ---------

    default_variable : number, list or np.ndarray : default Transfer_DEFAULT_BIAS
        specifies the input to the Mechanism to use if none is provided in a call to its
        `execute <Mechanism_Base.execute>` or `run <Mechanism_Base.run>` method;
        also serves as a template to specify the length of `variable <TransferMechanism.variable>` for
        `function <TransferMechanism.function>`, and the `primary outputState <OutputState_Primary>`
        of the Mechanism.

    size : int, list or np.ndarray of ints
        specifies default_variable as array(s) of zeros if **default_variable** is not passed as an argument;
        if **default_variable** is specified, it takes precedence over the specification of **size**.

    function : TransferFunction : default Linear
        specifies the function used to transform the input;  can be `Linear`, `Logistic`, `Exponential`,
        or a custom function.

    initial_value :  value, list or np.ndarray : default Transfer_DEFAULT_BIAS
        specifies the starting value for time-averaged input (only relevant if `integrator_mode
        <TransferMechanism.integrator_mode>` is True).
        COMMENT:
            Transfer_DEFAULT_BIAS SHOULD RESOLVE TO A VALUE
        COMMENT

    noise : float or function : default 0.0
        a stochastically-sampled value added to the result of the `function <TransferMechanism.function>`:
        if it is a float, it must be in the interval [0,1] and is used to scale the variance of a zero-mean Gaussian;
        if it is a function, it must return a scalar value.

    time_constant : float : default 1.0
        the time constant for exponential time averaging of input when the Mechanism is executed with `integrator_mode`
        set to True::

         result = (time_constant * current input) + ((1-time_constant) * result on previous time_step)

    range : Optional[Tuple[float, float]]
        specifies the allowable range for the result of `function <TransferMechanism.function>`:
        the first item specifies the minimum allowable value of the result, and the second its maximum allowable value;
        any element of the result that exceeds the specified minimum or maximum value is set to the value of
        `range <TransferMechanism.range>` that it exceeds.

    params : Optional[Dict[param keyword, param value]]
        a `parameter dictionary <ParameterState_Specification>` that can be used to specify the parameters for
        the Mechanism, its function, and/or a custom function and its parameters.  Values specified for parameters in
        the dictionary override any assigned to those parameters in arguments of the constructor.

    name : str : default TransferMechanism-<index>
        a string used for the name of the Mechanism.
        If not is specified, a default is assigned by `MechanismRegistry`
        (see :doc:`Registry <LINK>` for conventions used in naming, including for default and duplicate names).

    prefs : Optional[PreferenceSet or specification dict : Mechanism.classPreferences]
        the `PreferenceSet` for Mechanism.
        If it is not specified, a default is assigned using `classPreferences` defined in __init__.py
        (see :doc:`PreferenceSet <LINK>` for details).

    context : str : default componentType+INITIALIZING
        string used for contextualization of instantiation, hierarchical calls, executions, etc.

    Returns
    -------
    instance of TransferMechanism : TransferMechanism


    Attributes
    ----------

    variable : value: default Transfer_DEFAULT_BIAS
        the input to Mechanism's `function <TransferMechanism.function>`.
        COMMENT:
            :py:data:`Transfer_DEFAULT_BIAS <LINK->SHOULD RESOLVE TO VALUE>`
        COMMENT

    function : Function :  default Linear
        the Function used to transform the input.

    COMMENT:
       THE FOLLOWING IS THE CURRENT ASSIGNMENT
    COMMENT
    initial_value :  value, list or np.ndarray : Transfer_DEFAULT_BIAS
        specifies the starting value for time-averaged input (only relevant if `integrator_mode
        <TransferMechanism.integrator_mode>` is True and `time_constant <TransferMechanism.time_constant>` is not 1.0).
        COMMENT:
            Transfer_DEFAULT_BIAS SHOULD RESOLVE TO A VALUE
        COMMENT

    noise : float or function : default 0.0
        a stochastically-sampled value added to the output of the `function <TransferMechanism.function>`:
        if it is a float, it must be in the interval [0,1] and is used to scale the variance of a zero-mean Gaussian;
        if it is a function, it must return a scalar value.

    time_constant : float : default 1.0
        the time constant for exponential time averaging of input when the Mechanism is executed with `integrator_mode`
        set to True::

          result = (time_constant * current input) + ( (1-time_constant) * result on previous time_step)

    integrator_mode : boolean : default False
        when set to True, the Mechanism time averages its input according to an exponentially weighted moving average
        (see `time_constant <TransferMechanisms.time_constant>`).

    range : Optional[Tuple[float, float]]
        determines the allowable range of the result: the first value specifies the minimum allowable value
        and the second the maximum allowable value;  any element of the result that exceeds minimum or maximum
        is set to the value of `range <TransferMechanism.range>` it exceeds.  If `function <TransferMechanism.function>`
        is `Logistic`, `range <TransferMechanism.range>` is set by default to (0,1).

    value : 2d np.array [array(float64)]
        result of executing `function <TransferMechanism.function>`.

    previous_value : float
        the `value <TransferMechanism.value>` on the previous execution of the Mechanism.

    delta : float
        the change in `value <TransferMechanism.value>` from the previous execution of the Mechanism
        (i.e., `value <TransferMechanism.value>` - `previous_value <TransferMechanism.previous_value>`).

    output_states : *ContentAddressableList[OutputState]* : default [`RESULT <TRANSFER_MECHANISM_RESULT>`]
        list of Mechanism's `OutputStates <OutputStates>`.  By default there is a single OutputState,
        `RESULT <TRANSFER_MECHANISM_RESULT>`, that contains the result of a call to the Mechanism's
        `function <TransferMechanism.function>`;  additional `standard <TransferMechanism_Standard_OutputStates>`
        and/or custom OutputStates may be included, based on the specifications made in the **output_states** argument
        of the Mechanism's constructor.

    output_values : List[array(float64)]
        each item is the `value <OutputState.value>` of the corresponding OutputState in `output_states
        <TransferMechanism.output_states>`.  The default is a single item containing the result of the
        TransferMechanism's `function <TransferMechanism.function>`;  additional
        ones may be included, based on the specifications made in the
        **output_states** argument of the Mechanism's constructor (see `TransferMechanism Standard OutputStates
        <TransferMechanism_Standard_OutputStates>`).

    name : str : default TransferMechanism-<index>
        the name of the Mechanism.
        Specified in the **name** argument of the constructor for the Projection;
        if not specified, a default is assigned by `MechanismRegistry`
        (see :doc:`Registry <LINK>` for conventions used in naming, including for default and duplicate names).

    prefs : PreferenceSet or specification dict : Mechanism.classPreferences
        the `PreferenceSet` for Mechanism.
        Specified in the **prefs** argument of the constructor for the Mechanism;
        if it is not specified, a default is assigned using `classPreferences` defined in ``__init__.py``
        (see :doc:`PreferenceSet <LINK>` for details).

    """

    componentType = TRANSFER_MECHANISM

    classPreferenceLevel = PreferenceLevel.SUBTYPE
    # These will override those specified in TypeDefaultPreferences
    classPreferences = {
        kwPreferenceSetName: 'TransferCustomClassPreferences',
        kpReportOutputPref: PreferenceEntry(False, PreferenceLevel.INSTANCE),
        kpRuntimeParamStickyAssignmentPref: PreferenceEntry(False, PreferenceLevel.INSTANCE)
    }

    # TransferMechanism parameter and control signal assignments):
    paramClassDefaults = ProcessingMechanism_Base.paramClassDefaults.copy()
    paramClassDefaults.update({NOISE: None})

    standard_output_states = standard_output_states.copy()

    class ClassDefaults(ProcessingMechanism_Base.ClassDefaults):
        variable = [[0]]

    @tc.typecheck
    def __init__(self,
                 default_variable=None,
                 size=None,
                 input_states:tc.optional(tc.any(list, dict, Mechanism, OutputState, InputState))=None,
                 function=Linear,
                 initial_value=None,
                 noise=0.0,
                 time_constant=1.0,
                 integrator_mode=False,
                 range=None,
                 output_states:tc.optional(tc.any(list, dict))=[RESULT],
                 time_scale=TimeScale.TRIAL,
                 params=None,
                 name=None,
                 prefs:is_pref_set=None,
                 context=componentType+INITIALIZING):
        """Assign type-level preferences and call super.__init__
        """
        # MODIFIED 7/21/17 CW: Removed output_states = [RESULT] from initialization, due to potential bugs with
        # mutable default arguments (see: bit.ly/2uID3s3)
        if output_states is None:
            output_states = [RESULT]

        if default_variable is None and size is None:
            default_variable = [[0]]

        def nested_len(x):
            try:
                return sum(nested_len(y) for y in x)
            except:
                return 1
        self._variable_length = nested_len(default_variable)

        params = self._assign_args_to_param_dicts(function=function,
                                                  initial_value=initial_value,
                                                  input_states=input_states,
                                                  output_states=output_states,
                                                  noise=noise,
                                                  time_constant=time_constant,
                                                  integrator_mode=integrator_mode,
                                                  time_scale=time_scale,
                                                  range=range,
                                                  params=params)

        self.integrator_function = None

        if not isinstance(self.standard_output_states, StandardOutputStates):
            self.standard_output_states = StandardOutputStates(self,
                                                               self.standard_output_states,
                                                               indices=PRIMARY_OUTPUT_STATE)

        super(TransferMechanism, self).__init__(variable=default_variable,
                                                size=size,
                                                params=params,
                                                name=name,
                                                prefs=prefs,
                                                context=self)
        self.nv_state = None

    def _validate_params(self, request_set, target_set=None, context=None):
        """Validate FUNCTION and Mechanism params

        """

        super()._validate_params(request_set=request_set, target_set=target_set, context=context)

        # Validate FUNCTION
        if FUNCTION in target_set:
            transfer_function = target_set[FUNCTION]
            # FUNCTION is a Function
            if isinstance(transfer_function, Component):
                transfer_function_class = transfer_function.__class__
                transfer_function_name = transfer_function.__class__.__name__
            # FUNCTION is a function or method
            elif isinstance(transfer_function, (function_type, method_type)):
                transfer_function_class = transfer_function.__self__.__class__
                transfer_function_name = transfer_function.__self__.__class__.__name__
            # FUNCTION is a class
            elif inspect.isclass(transfer_function):
                transfer_function_class = transfer_function
                transfer_function_name = transfer_function.__name__

            if not transfer_function_class.componentType is TRANSFER_FUNCTION_TYPE:
                raise TransferError("Function {} specified as FUNCTION param of {} must be a {}".
                                    format(transfer_function_name, self.name, TRANSFER_FUNCTION_TYPE))

        # Validate INITIAL_VALUE
        if INITIAL_VALUE in target_set:
            initial_value = target_set[INITIAL_VALUE]
            if initial_value is not None:
                if not iscompatible(initial_value, self.instance_defaults.variable):
                    raise Exception(
                        "initial_value is {}, type {}\nself.instance_defaults.variable is {}, type {}".format(
                            initial_value,
                            type(initial_value),
                            self.instance_defaults.variable,
                            type(self.instance_defaults.variable),
                        )
                    )
                    raise TransferError(
                        "The format of the initial_value parameter for {} ({}) must match its input ({})".format(
                            append_type_to_name(self),
                            initial_value,
                            self.instance_defaults.variable[0],
                        )
                    )

        # FIX: SHOULD THIS (AND TIME_CONSTANT) JUST BE VALIDATED BY INTEGRATOR FUNCTION NOW THAT THEY ARE PROPERTIES??
        # Validate NOISE:
        if NOISE in target_set:
            self._validate_noise(target_set[NOISE], self.instance_defaults.variable)

        # Validate TIME_CONSTANT:
        if TIME_CONSTANT in target_set:
            time_constant = target_set[TIME_CONSTANT]
            if (not (isinstance(time_constant, float) and 0 <= time_constant <= 1)) and (time_constant != None):
                raise TransferError("time_constant parameter ({}) for {} must be a float between 0 and 1".
                                    format(time_constant, self.name))

        # Validate RANGE:
        if RANGE in target_set:
            range = target_set[RANGE]
            if range:
                if not (isinstance(range, tuple) and len(range)==2 and all(isinstance(i, numbers.Number) for i in range)):
                    raise TransferError("range parameter ({}) for {} must be a tuple with two numbers".
                                        format(range, self.name))
                if not range[0] < range[1]:
                    raise TransferError("The first item of the range parameter ({}) must be less than the second".
                                        format(range, self.name))

        # self.integrator_function = Integrator(
        #     # default_variable=self.default_variable,
        #                                       initializer = self.instance_defaults.variable,
        #                                       noise = self.noise,
        #                                       rate = self.time_constant,
        #                                       integration_type= ADAPTIVE)

    def _validate_noise(self, noise, var):
        # Noise is a list or array
        if isinstance(noise, (np.ndarray, list)):
            # Variable is a list/array
            if isinstance(var, (np.ndarray, list)):
                if len(noise) != np.array(var).size:
                    # Formatting noise for proper display in error message
                    try:
                        formatted_noise = list(map(lambda x: x.__qualname__, noise))
                    except AttributeError:
                        formatted_noise = noise
                    raise MechanismError(
                        "The length ({}) of the array specified for the noise parameter ({}) of {} "
                        "must match the length ({}) of the default input ({}). If noise is specified as"
                        " an array or list, it must be of the same size as the input."
                        .format(len(noise), formatted_noise, self.name, np.array(var).size,
                                var))
                else:
                    for noise_item in noise:
                        if not isinstance(noise_item, (float, int)) and not callable(noise_item):
                            raise MechanismError(
                                "The elements of a noise list or array must be floats or functions.")


            # Variable is not a list/array
            else:
                raise MechanismError("The noise parameter ({}) for {} may only be a list or array if the "
                                    "default input value is also a list or array.".format(noise, self.name))

            # # Elements of list/array have different types
            # if not all(isinstance(x, type(noise[0])) for x in noise):
            #     raise MechanismError("All elements of noise list/array ({}) for {} must be of the same type. "
            #                         .format(noise, self.name))

        elif not isinstance(noise, (float, int)) and not callable(noise):
            raise MechanismError(
                "Noise parameter ({}) for {} must be a float, function, or array/list of these."
                    .format(noise, self.name))

    def _try_execute_param(self, param, var):

        # param is a list; if any element is callable, execute it
        if isinstance(param, (np.ndarray, list)):
            for i in range(len(param)):
                if callable(param[i]):
                    param[i] = param[i]()
        # param is one function
        elif callable(param):
            # if the variable is a list/array, execute the param function separately for each element
            if isinstance(var, (np.ndarray, list)):
                if isinstance(var[0], (np.ndarray, list)):
                    new_param = []
                    for i in var[0]:
                        new_param.append(param())
                    param = new_param
                else:
                    new_param = []
                    for i in var:
                        new_param.append(param())
                    param = new_param
            # if the variable is not a list/array, execute the param function
            else:
                param = param()
        return param

    def _instantiate_parameter_states(self, context=None):

        from psyneulink.components.functions.function import Logistic
        # If function is a logistic, and range has not been specified, bound it between 0 and 1
        if ((isinstance(self.function, Logistic) or
                 (inspect.isclass(self.function) and issubclass(self.function,Logistic))) and
                self.range is None):
            self.range = (0,1)

        super()._instantiate_parameter_states(context=context)

    def _instantiate_attributes_before_function(self, context=None):

        super()._instantiate_attributes_before_function(context=context)

        if self.initial_value is None:
            self.initial_value = self.instance_defaults.variable

    def get_param_struct_type(self):
        with pnlvm.LLVMBuilderContext() as ctx:
            param_type_list = [self.function_object.get_param_struct_type()]
            if self.integrator_mode:
                assert self.integrator_function is not None
                param_type_list.append(self.integrator_function.get_param_struct_type())

            return ir.LiteralStructType(param_type_list)


    def get_context_struct_type(self):
        with pnlvm.LLVMBuilderContext() as ctx:
            context_type_list = [self.function_object.get_context_struct_type()]
            if self.integrator_mode:
                assert self.integrator_function is not None
                context_type_list.append(self.integrator_function.get_context_struct_type())

            context_type = ir.LiteralStructType(context_type_list)
            return context_type


    def get_output_struct_type(self):
        with pnlvm.LLVMBuilderContext() as ctx:
            vec_ty = ir.ArrayType(ctx.float_ty, self._variable_length)
            output_type = ir.LiteralStructType([vec_ty])
            return output_type


    def get_input_struct_type(self):
        with pnlvm.LLVMBuilderContext() as ctx:
            vec_ty = ir.ArrayType(ctx.float_ty, self._variable_length)
            input_type = ir.LiteralStructType([vec_ty])
            return input_type


    def __gen_llvm_clamp(self, builder, index, ctx, vo, min_val, max_val):
        ptri = builder.gep(vo, [ctx.int32_ty(0), index])
        ptro = builder.gep(vo, [ctx.int32_ty(0), index])

        val = builder.load(ptri)
        val = pnlvm.helpers.fclamp_const(builder, val, min_val, max_val)

        builder.store(val, ptro)
        pass

    def _gen_llvm_function(self):
        func_name = None
        llvm_func = None
        with pnlvm.LLVMBuilderContext() as ctx:
            func_ty = ir.FunctionType(ir.VoidType(),
                (self.get_param_struct_type().as_pointer(),
                 self.get_context_struct_type().as_pointer(),
                 self.get_input_struct_type().as_pointer(),
                 self.get_output_struct_type().as_pointer()))

            func_name = ctx.module.get_unique_name("integrator_machanism")
            llvm_func = ir.Function(ctx.module, func_ty, name=func_name)
            params, state, si, so = llvm_func.args
            for p in params, state, si, so:
                p.attributes.add('nonnull')
                p.attributes.add('noalias')

            # Create entry block
            block = llvm_func.append_basic_block(name="entry")
            builder = ir.IRBuilder(block)
            vi = builder.gep(si, [ctx.int32_ty(0), ctx.int32_ty(0)])
            vo = builder.gep(so, [ctx.int32_ty(0), ctx.int32_ty(0)])

            main_function = ctx.get_llvm_function(self.function_object.llvmSymbolName)
            mf_params = builder.gep(params, [ctx.int32_ty(0), ctx.int32_ty(0)])

            if self.integrator_mode:
                assert self.integrator_function is not None
                integrator_function = ctx.get_llvm_function(self.integrator_function.llvmSymbolName)
                output_param = integrator_function.args[3]
                vtmp = builder.alloca(output_param.type.gep(ctx.int32_ty(0)), 1)
                if_params = builder.gep(params, [ctx.int32_ty(0), ctx.int32_ty(1)])
                if_state = builder.gep(state, [ctx.int32_ty(0), ctx.int32_ty(1)])
                builder.call(integrator_function, [if_params, if_state, vi, vtmp])

                # We know that TransferFunction is not stateful, but a consistent interface would be nicer
                builder.call(main_function, [mf_params, vtmp, vo])
            else:
                # We know that TransferFunction is not stateful, but a consistent interface would be nicer
                builder.call(main_function, [mf_params, vi, vo])

            if self.range is not None:
                kwargs = {"ctx":ctx, "vo":vo, "min_val":self.range[0], "max_val":self.range[1]}
                inner = functools.partial(self.__gen_llvm_clamp, **kwargs)

                vector_length = ctx.int32_ty(self._variable_length)
                builder = pnlvm.helpers.for_loop_zero_inc(builder, vector_length, inner, "linear")

            builder.ret_void()
        return func_name


    def _bin_execute(self,
                 variable=None,
                 runtime_params=None,
                 clock=CentralClock,
                 time_scale=TimeScale.TRIAL,
                 context=None):

        transfer_params = self.function_object.get_param_initializer()

        if self.integrator_mode:
            assert self.integrator_function is not None
            integrator_params = self.integrator_function.get_param_initializer()
        else:
            integrator_params = tuple()

        bf = self._llvmBinFunction

        def nested_len(x):
            try:
                return sum(nested_len(y) for y in x)
            except:
                return 1
        ret = np.zeros(nested_len(variable)) #default is numpy.float64
        par_struct_ty, context_struct_ty, vi_ty, vo_ty = bf.byref_arg_types

        if self.nv_state is None:
            initializer_t = self.function_object.get_context_initializer()
            initializer_i = self.integrator_function.get_context_initializer()
            self.nv_state = context_struct_ty(initializer_t, initializer_i)
        ct_context = self.nv_state
        ct_param = par_struct_ty(transfer_params, integrator_params)

        variable = np.asarray(variable, dtype=np.float64)
        # This is bit hacky because numpy can't cast to arrays
        ct_vi = variable.ctypes.data_as(ctypes.POINTER(vi_ty))
        ct_vo = ret.ctypes.data_as(ctypes.POINTER(vo_ty))

        bf(ct_param, ct_context, ct_vi, ct_vo)

        return ret


    def _execute(self,
                 variable=None,
                 runtime_params=None,
                 clock=CentralClock,
                 time_scale=TimeScale.TRIAL,
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
        - context (str)

        Returns the following values in self.value (2D np.array) and in
            the value of the corresponding outputState in the self.output_states list:
            - activation value (float)
            - mean activation value (float)
            - standard deviation of activation values (float)

        :param self:
        :param variable (float)
        :param params: (dict)
        :param context: (str)
        :rtype self.outputState.value: (number)
        """

        # FIX: ??CALL check_args()??

        # FIX: IS THIS CORRECT?  SHOULD THIS BE SET TO INITIAL_VALUE
        # FIX:     WHICH SHOULD BE DEFAULTED TO 0.0??
        # Use self.instance_defaults.variable to initialize state of input

        # FIX: NEED TO GET THIS TO WORK WITH CALL TO METHOD:
        time_scale = self.time_scale
        integrator_mode = self.integrator_mode

        #region ASSIGN PARAMETER VALUES

        time_constant = self.time_constant
        range = self.range
        noise = self.noise

        #endregion

        #region EXECUTE TransferMechanism FUNCTION ---------------------------------------------------------------------

        # FIX: NOT UPDATING self.previous_input CORRECTLY
        # FIX: SHOULD UPDATE PARAMS PASSED TO integrator_function WITH ANY RUNTIME PARAMS THAT ARE RELEVANT TO IT

        # Update according to time-scale of integration
        if integrator_mode:
        # if time_scale is TimeScale.TIME_STEP:

            if not self.integrator_function:

                self.integrator_function = AdaptiveIntegrator(
                                            variable,
                                            initializer = self.initial_value,
                                            noise = self.noise,
                                            rate = self.time_constant,
                                            owner = self)

            current_input = self.integrator_function.execute(variable,
                                                        # Should we handle runtime params?
                                                              params={INITIALIZER: self.initial_value,
                                                                      NOISE: self.noise,
                                                                      RATE: self.time_constant},
                                                              context=context

                                                             )[0]
        else:
        # elif time_scale is TimeScale.TRIAL:
            noise = self._try_execute_param(self.noise, variable)
            # formerly: current_input = self.input_state.value + noise
            # (MODIFIED 7/13/17 CW) this if/else below is hacky: just allows a nicer error message
            # when the input is given as a string.
            if (np.array(noise) != 0).any():
                current_input = variable[0] + noise
            else:

                current_input = self.variable[0]

        # self.previous_input = current_input

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

    def _report_mechanism_execution(self, input, params, output):
        """Override super to report previous_input rather than input, and selected params
        """
        # KAM Changed 8/29/17 print_input = self.previous_input --> print_input = input
        # because self.previous_input is not a valid attrib of TransferMechanism

        print_input = input
        print_params = params.copy()
        # Only report time_constant if in TIME_STEP mode
        if params['time_scale'] is TimeScale.TRIAL:
            del print_params[TIME_CONSTANT]
        # Suppress reporting of range (not currently used)
        del print_params[RANGE]

        super()._report_mechanism_execution(input_val=print_input, params=print_params)


    # def terminate_function(self, context=None):
    #     """Terminate the process
    #
    #     called by process.terminate() - MUST BE OVERRIDDEN BY SUBCLASS IMPLEMENTATION
    #     returns output
    #
    #     :rtype CurrentStateTuple(state, confidence, duration, controlModulatedParamValues)
    #     """
    #     # IMPLEMENTATION NOTE:  TBI when time_step is implemented for TransferMechanism
    #
    @property
    def range(self):
        return self._range


    @range.setter
    def range(self, value):
        self._range = value

    # MODIFIED 4/17/17 NEW:
    @property
    def noise (self):
        return self._noise

    @noise.setter
    def noise(self, value):
        self._noise = value

    @property
    def time_constant(self):
        return self._time_constant

    @time_constant.setter
    def time_constant(self, value):
        self._time_constant = value
    # # MODIFIED 4/17/17 END

    @property
    def previous_value(self):
        if self.integrator_function:
            return self.integrator_function.previous_value
        return None

    @property
    def delta(self):
        if self.integrator_function:
            return self.value - self.integrator_function.previous_value
        return None