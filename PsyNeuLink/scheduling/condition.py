# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.


# ********************************************* Condition ***************************************************************

"""

COMMENT:
    [**NAME OF THE MODULE SHOULD BE CAPITALIZED]
    K: Agreed, along with scheduling folder to match the project style
    JDC: Yep
COMMENT

.. _Condition_Overview

Overview
--------

`Conditions <Condition>` are used to specify when `Mechanisms <Mechanism>` execute.  They fall broadly into two
categories: *absolute* ones that specify the behavior of a Mechanism irrespective of others (e.g., the exact number
of times it should be executed, or whether this should be based on its own `value <Mechanism.value>`);
COMMENT:
K: I'm not sure what that means
JDC: A CRITERION LIKE THE minimal_change ONE IN THE EXAMPLE WE JUST WORKED OUT;  OK?
COMMENT
and *relative* ones, that specify whether and how its execution depends on other Mechanisms (e.g., the frequency with
which it executes relative to others, or that it begin executing and/or execute repeatedly until a Condition is met
for some other Mechanism).  Each Condition is associated with an `owner <Condition.owner>` (a `Mechanism` to which the
Condition belongs), and a `scheduler <Condition.scheduler>` that maintains most of the data required to test for
satisfaction of the condition.

.. _Condition_Creation:

Creating Conditions
-----------------------

Conditions can be created at any time, and take effect immediately for the execution of any `Scheduler(s) <Scheduler>`
with which they are associated.  The Condition's **dependencies** and **func** arguments must both be explicitly
specified.  These are used to determine whether a Condition is satisfied during each `round of execution <LINK>`:
COMMENT:
K: round of execution is poorly defined and should refer to a TimeScale)
JDC:  RIGHT;  WE SHOULD SETTLE ON PASS VS. ROUND OF EXECUTION, AND THEN DO THE APPROPRIATE SEARCH AND REPLACE
COMMENT
`func <Condition.func>` is called with `dependencies <Condition.dependencies>` as its parameter
COMMENT:
JDC: "additonal named and unnamed arguments" SEEMS VAGUE;  FOR WHAT?  EXAMPLE MIGHT HELP?
COMMENT
(and optionally, additional named and unnamed arguments).
COMMENT:
     [**??func AND dependencies NEED TO BE CLARIFIED:  WHAT FORMAT, EXAMPLE OF HOW THEY WORK??]
    K: It's explained in the previous version
             Each Condition must
            - be a subclass of `Condition`<Condition>
            - pass `dependencies` as the first argument to the __init__ function of Condition
            - pass `func` as the second argument to the __init__ function of Condition

        In determining whether a Condition is satisfied, `func` is called with `dependencies` as parameter (and optionally,
        additional named and unnamed arguments).
    They are not in an exact format by design, because they can be customized by any advanced user.
    JDC: SHOULD DISCUSS;  I'M NOT SURE I FULLY UNDERSTAND
COMMENT

Hint:
    If you do not want to use the dependencies parameter, and instead want to use only args or kwargs, you may
    pass a dummy variable for dependencies. See `AfterNCallsCombined`<AfterNCallsCombined> for reference:

    class AfterNCallsCombined(Condition):
        def __init__(self, *dependencies, n=None, time_scale=TimeScale.TRIAL):
            def func(_none, *dependencies, n=None):
                if self.scheduler is None:
                    raise ConditionError('{0}: self.scheduler is None - scheduler must be assigned'.format(type(self).__name__))
                if n is None:
                    raise ConditionError('{0}: keyword argument n is None'.format(type(self).__name__))
                count_sum = 0
                for d in dependencies:
                    count_sum += self.scheduler.counts_total[time_scale][d]
                return count_sum >= n
            super().__init__(None, func, *dependencies, n=n)

.. Condition_Structure:

Structure
---------

COMMENT:
     **??DESCRIBE HOW CONDITIONS ARE STRUCTURED;
     INCLUDE FULL LIST OF CONDITIONS
    K: they are listed in the bottom of the Condition doc page by nature of being classes in this file,
     at least on the version I see
    JDC: GOT IT.  THANKS. THAT SAID, IN OTHER PARTS OF THE DOCUMENTION (AND AT THE RISK OF A BIT OF REDUNDANCY)
         I'VE INCLUDED BRIEF SUMMARIIES OF ATTRIBUTES AND/OR METHODS IN THE DOSCSTRING OF THE MAIN CLASS,
         WHICH I THINK MAY BE EASIER TO READ.  I'LL ADD HERE, AND WE CAN REMOVE IF IT SEEMS OVERLY REDUNDANT.
COMMENT

Each type of Condition is a subclass of Condition.  The following types are provided:

Absolute Conditions:
    * `BeforePass`
    * `AtPass`
    * `AfterPass`
    * `AfterNPasses`
    * `EveryNPasses`
    * `BeforeTrial`
    * `AfterTrial`
    * `AfterNTrials`
    * `BeforeNCalls`
    * `AtNCalls`
    * `AfterCall`
    * `AfterNCalls`
    * `AfterNCallsCombined`
    * `EveryNCalls`
    * `JustRan`
    * `AllHaveRun`
    * `WhenFinished`
    * `WhenFinishedAny`
    * `WhenFinishedAll`
    * `Always`
    * `Never`

Relative Conditions:
    * `All`
    * `Any`
    * `Not`

.. Condition_Execution:

Execution
---------

A Condition is evaluated when a `Scheduler` is run, by calling the Condition's `is_satisfied` method.  If it returns
`True` then the `Component` associated with the Condition is executed.
COMMENT:
     **??DESCRIBE HOW CONITION IS EVALUATED
    K: It doesn't "execute" exactly. A condition is satisfied when its is_satisfied function returns True, and is_satisfied
    is called when the scheduler runs
    JDC: IS THE ABOVE BETTER?
         IS THE SATISFACTION OF A CONDTIION ALWAYS ASSOCIATED WITH THE EXECUTION OF A COMPONENT?
         JUST *ONE* COMPONENT
COMMENT

.. _Condition_Class_Reference

Class Reference
---------------

"""

import logging

from PsyNeuLink.Globals.TimeScale import TimeScale

logger = logging.getLogger(__name__)


class ConditionError(Exception):
    def __init__(self, error_value):
        self.error_value = error_value

    def __str__(self):
        return repr(self.error_value)


class ConditionSet(object):
    """
    An object used in conjunction with `Scheduler`<Scheduler> to store all the associated conditions with their owners.
    """
    def __init__(self, scheduler=None, conditions=None):
        """
        :param self:
        :param scheduler: a :keyword:`Scheduler` that these conditions are associated with, which maintains any state necessary for these conditions
        :param conditions: a :keyword:`dict` mapping :keyword:`Component`s to :keyword:`iterable`s of :keyword:`Condition`s, can be added later with :keyword:`add_condition`
        """
        self.conditions = conditions if conditions is not None else {}
        self.scheduler = scheduler

    def __contains__(self, item):
        return item in self.conditions

    @property
    def scheduler(self):
        return self._scheduler

    @scheduler.setter
    def scheduler(self, value):
        logger.debug('ConditionSet ({0}) setting scheduler to {1}'.format(type(self).__name__, value))
        self._scheduler = value

        for owner, cond in self.conditions.items():
            cond.scheduler = value

    def add_condition(self, owner, condition):
        """
        :param self:
        :param owner: the :keyword:`Component` that is dependent on the :param conditions:
        :param conditions: a :keyword:`Condition` (including All or Any)
        """
        logger.debug('add_condition: Setting scheduler of {0}, (owner {2}) to self.scheduler ({1})'.format(condition, self.scheduler, owner))
        condition.owner = owner
        condition.scheduler = self.scheduler
        self.conditions[owner] = condition

    def add_condition_set(self, conditions):
        """
        :param self:
        :param conditions: a :keyword:`dict` mapping :keyword:`Component`s to :keyword:`Condition`s, can be added later with :keyword:`add_condition`
        """
        for owner in conditions:
            conditions[owner].owner = owner
            conditions[owner].scheduler = self.scheduler
            self.conditions[owner] = conditions[owner]


class Condition(object):
    """
    Used in conjunction with a `Scheduler` to specify the pattern of execution for `Mechanisms <Mechanism>` in a
    `System.

    Arguments
    ---------

    dependencies : **??LIST? DICT?
        K: ANYTHING
        JDC: NOT SURE I UNDERSTAND.  THERE IS NO SYNTAX FOR SPECIFICATION OF DEPENDENCIES?
             WHAT IF JUST AN INTEGER IS GIVEN?
        not just mechanisms, they can be anything at all
        one or more `Mechanisms <Mechanism>` over which `func <Condition.func>` is evaluated to determine satisfaction
        of the `Condition`;  user must ensure that dependencies are suitable as func parameters
    func : function
        JDC: **??FORMAT?
        K: probably below in my version was an accident. func is just any function
        JDC: OK AS CORRECTED ABOVE?
        func is evaluated to determine satisfaction of the `Condition`

    args :
        additional formal arguments passed to func

    kwargs :
        additional keyword arguments passed to func

    Attributes
    ----------

    scheduler : Scheduler
        the `Scheduler` with which the Condition is associated;  the Scheduler's state is used to evaluate whether
        the Condition`s specifications are satisfied.

    owner (Component):
        the `Component` with which the Condition is associated, and the execution of which it determines.

        """
    def __init__(self, dependencies, func, *args, **kwargs):
        """
        :param self:
        :param dependencies: one or more PNL objects over which func is evaluated to determine satisfaction of the :keyword:`Condition`
            user must ensure that dependencies are suitable as func parameters
        :param func: func is evaluated to determine satisfaction of the :keyword:`Condition`
        :param args: additional formal arguments passed to func
        :param kwargs: additional keyword arguments passed to func
        """
        self.dependencies = dependencies
        self.func = func
        self.args = args
        self.kwargs = kwargs

        self._scheduler = None
        self._owner = None
        # logger.debug('{1} dependencies: {0}'.format(dependencies, type(self).__name__))

    @property
    def scheduler(self):
        return self._scheduler

    @scheduler.setter
    def scheduler(self, value):
        logger.debug('Condition ({0}) setting scheduler to {1}'.format(type(self).__name__, value))
        self._scheduler = value

    @property
    def owner(self):
        return self._owner

    @owner.setter
    def owner(self, value):
        logger.debug('Condition ({0}) setting owner to {1}'.format(type(self).__name__, value))
        self._owner = value

    def is_satisfied(self):
        logger.debug('Condition ({0}) using scheduler {1}'.format(type(self).__name__, self.scheduler))
        has_args = len(self.args) > 0
        has_kwargs = len(self.kwargs) > 0

        if has_args and has_kwargs:
            return self.func(self.dependencies, *self.args, **self.kwargs)
        if has_args:
            return self.func(self.dependencies, *self.args)
        if has_kwargs:
            return self.func(self.dependencies, **self.kwargs)
        return self.func(self.dependencies)

#########################################################################################################
# Included Conditions
#########################################################################################################

######################################################################
# Static Conditions
#   - independent of components and time
######################################################################


class Always(Condition):
    """
    Always

    Parameters:

    Satisfied when:
        - always satisfied

    Notes:

    """
    def __init__(self):
        super().__init__(True, lambda x: x)


class Never(Condition):
    """
    Never

    Parameters:

    Satisfied when:
        - never satisfied

    Notes:

    """
    def __init__(self):
        super().__init__(False, lambda x: x)

######################################################################
# Relative Conditions
#   - based on other Conditions
######################################################################

# TODO: create this class to subclass All and Any from
# class CompositeCondition(Condition):
    # def


class All(Condition):
    """
    All

    Parameters:
        - args (argtuple): one or more :keyword:`Condition`s

    Satisfied when:
        - All args are satisfied

    Notes:
        To initialize with a list (for example),
            conditions = [AfterNCalls(mechanism, 5) for mechanism in mechanism_list]
        unpack the list to supply its members as args
            composite_condition = All(*conditions)
    """
    def __init__(self, *args):
        """
        :param self:
        :param args: one or more :keyword:`Condition`s, all of which must be satisfied to satisfy this composite condition
        """
        super().__init__(args, self.satis)

    @Condition.scheduler.setter
    def scheduler(self, value):
        for cond in self.dependencies:
            logger.debug('schedule setter: Setting scheduler of {0} to ({1})'.format(cond, value))
            if cond.scheduler is None:
                cond.scheduler = value

    @Condition.owner.setter
    def owner(self, value):
        for cond in self.dependencies:
            logger.debug('owner setter: Setting owner of {0} to ({1})'.format(cond, value))
            if cond.owner is None:
                cond.owner = value

    def satis(self, conds):
        for cond in conds:
            if not cond.is_satisfied():
                return False
        return True


class Any(Condition):
    """
    Any

    Parameters:
        - args: one or more :keyword:`Condition`s

    Satisfied when:
        - All args are satisfied

    Notes:
        To initialize with a list (for example), conditions = [AfterNCalls(mechanism, 5) for mechanism in mechanism_list], unpack the list to supply its members as args

        composite_condition = Any(*conditions)
    """
    def __init__(self, *args):
        """
        :param self:
        :param args: one or more :keyword:`Condition`s, any of which must be satisfied to satisfy this composite condition
        """
        super().__init__(args, self.satis)

    @Condition.scheduler.setter
    def scheduler(self, value):
        logger.debug('Any setter args: {0}'.format(self.dependencies))
        for cond in self.dependencies:
            logger.debug('schedule setter: Setting scheduler of {0} to ({1})'.format(cond, value))
            if cond.scheduler is None:
                cond.scheduler = value

    @Condition.owner.setter
    def owner(self, value):
        for cond in self.dependencies:
            logger.debug('owner setter: Setting owner of {0} to ({1})'.format(cond, value))
            if cond.owner is None:
                cond.owner = value

    def satis(self, conds):
        for cond in conds:
            if cond.is_satisfied():
                return True
        return False


class Not(Condition):
    """
    Not

    Parameters:
        - condition (Condition): a :keyword:`Condition`

    Satisfied when:
        - condition is not satisfied

    Notes:

    """
    def __init__(self, condition):
        super().__init__(condition, lambda c: not c.is_satisfied())

    @Condition.scheduler.setter
    def scheduler(self, value):
        self.dependencies.scheduler = value

    @Condition.owner.setter
    def owner(self, value):
        self.dependencies.owner = value

######################################################################
# Time-based Conditions
#   - satisfied based only on TimeScales
######################################################################


class BeforePass(Condition):
    """
    BeforePass

    Parameters:
        - n (int): the pass after which this condition will be satisfied
        - time_scale (TimeScale): the TimeScale used as basis for counting passes. Defaults to TimeScale.TRIAL

    Satisfied when:
        - within the scope of time_scale, at most n-1 passes have occurred

    Notes:
        Counts of TimeScales are zero-indexed (that is, the first Pass is pass 0, the second Pass is pass 1, etc.). So,
        BeforePass(2) is satisfied at pass 0 and pass 1

    """
    def __init__(self, n, time_scale=TimeScale.TRIAL):
        def func(n, time_scale):
            if self.scheduler is None:
                raise ConditionError('{0}: self.scheduler is None - scheduler must be assigned'.format(type(self).__name__))
            return self.scheduler.times[time_scale][TimeScale.PASS] < n
        super().__init__(n, func, time_scale)


class AtPass(Condition):
    """
    AtPass

    Parameters:
        - n (int): the pass at which this condition will be satisfied
        - time_scale (TimeScale): the TimeScale used as basis for counting passes. Defaults to TimeScale.TRIAL

    Satisfied when:
        - within the scope of time_scale, exactly n passes have occurred

    Notes:
        Counts of TimeScales are zero-indexed (that is, the first Pass is pass 0, the second Pass is pass 1, etc.). So,
        AtPass(1) is satisfied when one pass (pass 0) has already occurred.

    """
    def __init__(self, n, time_scale=TimeScale.TRIAL):
        def func(n):
            if self.scheduler is None:
                raise ConditionError('{0}: self.scheduler is None - scheduler must be assigned'.format(type(self).__name__))
            try:
                return self.scheduler.times[time_scale][TimeScale.PASS] == n
            except KeyError as e:
                raise ConditionError('{0}: {1}, is time_scale set correctly? Currently: {2}'.format(type(self).__name__, e, time_scale))
        super().__init__(n, func)


class AfterPass(Condition):
    """
    AfterPass

    Parameters:
        - n (int): the pass after which this condition will be satisfied
        - time_scale (TimeScale): the TimeScale used as basis for counting passes. Defaults to TimeScale.TRIAL

    Satisfied when:
        - within the scope of time_scale, at least n+1 passes have occurred

    Notes:
        Counts of TimeScales are zero-indexed (that is, the first Pass is pass 0, the second Pass is pass 1, etc.). So,
        AfterPass(1) is satisfied after pass 1 has occurred, at pass 2, pass 3, pass 4, etc.

    """
    def __init__(self, n, time_scale=TimeScale.TRIAL):
        def func(n, time_scale):
            if self.scheduler is None:
                raise ConditionError('{0}: self.scheduler is None - scheduler must be assigned'.format(type(self).__name__))
            return self.scheduler.times[time_scale][TimeScale.PASS] > n
        super().__init__(n, func, time_scale)


class AfterNPasses(Condition):
    """
    AfterPass

    Parameters:
        - n (int): the number of TimeScale.PASSes after which this condition will be satisfied
        - time_scale (TimeScale): the TimeScale used as basis for counting passes. Defaults to TimeScale.TRIAL

    Satisfied when:
        - the count of TimeScale.PASSes within time_scale is at least n

    """
    def __init__(self, n, time_scale=TimeScale.TRIAL):
        def func(n, time_scale):
            if self.scheduler is None:
                raise ConditionError('{0}: self.scheduler is None - scheduler must be assigned'.format(type(self).__name__))
            return self.scheduler.times[time_scale][TimeScale.PASS] >= n
        super().__init__(n, func, time_scale)


class EveryNPasses(Condition):
    """
    EveryNPasses

    Parameters:
        - n (int): the frequency of passes with which this condition will be satisfied
        - time_scale (TimeScale): the TimeScale used as basis for counting passes. Defaults to TimeScale.TRIAL

    Satisfied when:
        - the number of passes that has occurred within time_scale is evenly divisible by n

    Notes:
        All EveryNPasses conditions will be satisfied at pass 0

    """
    def __init__(self, n, time_scale=TimeScale.TRIAL):
        def func(n, time_scale):
            if self.scheduler is None:
                raise ConditionError('{0}: self.scheduler is None - scheduler must be assigned'.format(type(self).__name__))
            return self.scheduler.times[time_scale][TimeScale.PASS] % n == 0
        super().__init__(n, func, time_scale)


class BeforeTrial(Condition):
    """
    BeforeTrial

    Parameters:
        - n (int): the trial after which this condition will be satisfied
        - time_scale (TimeScale): the TimeScale used as basis for counting trials. Defaults to TimeScale.RUN

    Satisfied when:
        - within the scope of time_scale, at most n-1 trials have occurred

    Notes:
        Counts of TimeScales are zero-indexed (that is, the first Trial is trial 0, the second Trial is trial 1, etc.). So,
        BeforeTrial(2) is satisfied at trial 0 and trial 1

    """
    def __init__(self, n, time_scale=TimeScale.RUN):
        def func(n):
            if self.scheduler is None:
                raise ConditionError('{0}: self.scheduler is None - scheduler must be assigned'.format(type(self).__name__))
            try:
                return self.scheduler.times[time_scale][TimeScale.TRIAL] < n
            except KeyError as e:
                raise ConditionError('{0}: {1}, is time_scale set correctly? Currently: {2}'.format(type(self).__name__, e, time_scale))
        super().__init__(n, func)


class AtTrial(Condition):
    """
    AtTrial

    Parameters:
        - n (int): the trial at which this condition will be satisfied
        - time_scale (TimeScale): the TimeScale used as basis for counting trials. Defaults to TimeScale.RUN

    Satisfied when:
        - within the scope of time_scale, exactly n trials have occurred

    Notes:
        Counts of TimeScales are zero-indexed (that is, the first Trial is trial 0, the second Trial is trial 1, etc.). So,
        AtTrial(1) is satisfied when one trial (trial 0) has already occurred.

    """
    def __init__(self, n, time_scale=TimeScale.RUN):
        def func(n):
            if self.scheduler is None:
                raise ConditionError('{0}: self.scheduler is None - scheduler must be assigned'.format(type(self).__name__))
            try:
                return self.scheduler.times[time_scale][TimeScale.TRIAL] == n
            except KeyError as e:
                raise ConditionError('{0}: {1}, is time_scale set correctly? Currently: {2}'.format(type(self).__name__, e, time_scale))
        super().__init__(n, func)


class AfterTrial(Condition):
    """
    AfterTrial

    Parameters:
        - n (int): the trial after which this condition will be satisfied
        - time_scale (TimeScale): the TimeScale used as basis for counting trials. Defaults to TimeScale.RUN

    Satisfied when:
        - within the scope of time_scale, at least n+1 trials have occurred

    Notes:
        Counts of TimeScales are zero-indexed (that is, the first Trial is trial 0, the second Trial is trial 1, etc.). So,
        AfterTrial(1) is satisfied after trial 1 has occurred, at trial 2, trial 3, trial 4, etc.

    """
    def __init__(self, n, time_scale=TimeScale.RUN):
        def func(n):
            if self.scheduler is None:
                raise ConditionError('{0}: self.scheduler is None - scheduler must be assigned'.format(type(self).__name__))
            try:
                return self.scheduler.times[time_scale][TimeScale.TRIAL] > n
            except KeyError as e:
                raise ConditionError('{0}: {1}, is time_scale set correctly? Currently: {2}'.format(type(self).__name__, e, time_scale))
        super().__init__(n, func)


class AfterNTrials(Condition):
    """
    AfterNTrials

    Parameters:
        - n (int): the number of TimeScale.TRIALs after which this condition will be satisfied
        - time_scale (TimeScale): the TimeScale used as basis for counting trials. Defaults to TimeScale.RUN

    Satisfied when:
        - the count of TimeScale.TRIALs within time_scale is at least n

    Notes:

    """
    def __init__(self, n, time_scale=TimeScale.RUN):
        def func(n, time_scale):
            if self.scheduler is None:
                raise ConditionError('{0}: self.scheduler is None - scheduler must be assigned'.format(type(self).__name__))
            return self.scheduler.times[time_scale][TimeScale.TRIAL] >= n
        super().__init__(n, func, time_scale)

######################################################################
# Component-based Conditions
#   - satisfied based on executions or state of Components
######################################################################


class BeforeNCalls(Condition):
    """
    BeforeNCalls

    Parameters:
        - dependency (Component):
        - n (int): the number of executions of dependency at which this condition will be satisfied
        - time_scale (TimeScale): the TimeScale used as basis for counting executions of dependency. Defaults to TimeScale.TRIAL

    Satisfied when:
        - dependency has been executed exactly n times within the scope of time_scale

    Notes:

    """
    def __init__(self, dependency, n, time_scale=TimeScale.TRIAL):
        def func(dependency, n):
            if self.scheduler is None:
                raise ConditionError('{0}: self.scheduler is None - scheduler must be assigned'.format(type(self).__name__))
            num_calls = self.scheduler.counts_total[time_scale][dependency]
            logger.debug('{0} has reached {1} num_calls in {2}'.format(dependency, num_calls, time_scale.name))
            return num_calls < n
        super().__init__(dependency, func, n)

# NOTE:
# The behavior of AtNCalls is not desired (i.e. depending on the order mechanisms are checked, B running AtNCalls(A, x))
# may run on both the xth and x+1st call of A; if A and B are not parent-child
# A fix could invalidate key assumptions and affect many other conditions
# Since this condition is unlikely to be used, it's best to leave it for now


class AtNCalls(Condition):
    """
    AtNCalls

    Parameters:
        - dependency (Component):
        - n (int): the number of executions of dependency at which this condition will be satisfied
        - time_scale (TimeScale): the TimeScale used as basis for counting executions of dependency. Defaults to TimeScale.TRIAL

    Satisfied when:
        - dependency has been executed exactly n times within the scope of time_scale

    Notes:

    """
    def __init__(self, dependency, n, time_scale=TimeScale.TRIAL):
        def func(dependency, n):
            if self.scheduler is None:
                raise ConditionError('{0}: self.scheduler is None - scheduler must be assigned'.format(type(self).__name__))
            num_calls = self.scheduler.counts_total[time_scale][dependency]
            logger.debug('{0} has reached {1} num_calls in {2}'.format(dependency, num_calls, time_scale.name))
            return num_calls == n
        super().__init__(dependency, func, n)


class AfterCall(Condition):
    """
    AfterCall

    Parameters:
        - dependency (Component):
        - n (int): the number of executions of dependency after which this condition will be satisfied
        - time_scale (TimeScale): the TimeScale used as basis for counting executions of dependency. Defaults to TimeScale.TRIAL

    Satisfied when:
        - dependency has been executed at least n+1 times within the scope of time_scale

    Notes:

    """
    def __init__(self, dependency, n, time_scale=TimeScale.TRIAL):
        def func(dependency, n):
            if self.scheduler is None:
                raise ConditionError('{0}: self.scheduler is None - scheduler must be assigned'.format(type(self).__name__))
            num_calls = self.scheduler.counts_total[time_scale][dependency]
            logger.debug('{0} has reached {1} num_calls in {2}'.format(dependency, num_calls, time_scale.name))
            return num_calls > n
        super().__init__(dependency, func, n)


class AfterNCalls(Condition):
    """
    AfterNCalls

    Parameters:
        - dependency (Component):
        - n (int): the number of executions of dependency after which this condition will be satisfied
        - time_scale (TimeScale): the TimeScale used as basis for counting executions of dependency. Defaults to TimeScale.TRIAL

    Satisfied when:
        - dependency has been executed at least n+1 times within the scope of time_scale

    Notes:

    """
    def __init__(self, dependency, n, time_scale=TimeScale.TRIAL):
        def func(dependency, n):
            if self.scheduler is None:
                raise ConditionError('{0}: self.scheduler is None - scheduler must be assigned'.format(type(self).__name__))
            num_calls = self.scheduler.counts_total[time_scale][dependency]
            logger.debug('{0} has reached {1} num_calls in {2}'.format(dependency, num_calls, time_scale.name))
            return num_calls >= n
        super().__init__(dependency, func, n)


class AfterNCallsCombined(Condition):
    """
    AfterNCallsCombined

    Parameters:
        - *dependencies (Components): variable length
        - n (int): the number of executions of all dependencies after which this condition will be satisfied. Defaults to None
        - time_scale (TimeScale): the TimeScale used as basis for counting executions of dependencies. Defaults to TimeScale.TRIAL

    Satisfied when:
        - Among all dependencies, there have been at least n+1 executions within the scope of time_scale

    Notes:

    """
    def __init__(self, *dependencies, n=None, time_scale=TimeScale.TRIAL):
        logger.debug('{0} args: deps {1}, n {2}, ts {3}'.format(type(self).__name__, dependencies, n, time_scale))

        def func(_none, *dependencies, n=None):
            if self.scheduler is None:
                raise ConditionError('{0}: self.scheduler is None - scheduler must be assigned'.format(type(self).__name__))
            if n is None:
                raise ConditionError('{0}: keyword argument n is None'.format(type(self).__name__))
            count_sum = 0
            for d in dependencies:
                count_sum += self.scheduler.counts_total[time_scale][d]
                logger.debug('{0} has reached {1} num_calls in {2}'.format(d, self.scheduler.counts_total[time_scale][d], time_scale.name))
            return count_sum >= n
        super().__init__(None, func, *dependencies, n=n)


class EveryNCalls(Condition):
    """
    EveryNCalls

    Parameters:
        - dependency (Component):
        - n (int): the frequency of executions of dependency with which this condition will be satisfied

    Satisfied when:
        - since the last time this conditon's owner was called, the number of calls of dependency is at least n

    Notes:
        Whenever a Component is run, the Scheduler's count of each dependency that is "useable" by the Component is
        reset to 0

    """
    def __init__(self, dependency, n):
        def func(dependency, n):
            if self.scheduler is None:
                raise ConditionError('{0}: self.scheduler is None - scheduler must be assigned'.format(type(self).__name__))
            num_calls = self.scheduler.counts_useable[dependency][self.owner]
            logger.debug('{0} has reached {1} num_calls'.format(dependency, num_calls))
            return num_calls >= n
        super().__init__(dependency, func, n)


class JustRan(Condition):
    """
    JustRan

    Parameters:
        - dependency (Component):

    Satisfied when:
        - dependency has been run (or told to run) in the previous TimeScale.TIME_STEP

    Notes:
        This condition can transcend divisions between TimeScales. That is, if A runs in the final time step in a trial,
        JustRan(A) will be satisfied at the beginning of the next trial.

    """
    def __init__(self, dependency):
        def func(dependency):
            if self.scheduler is None:
                raise ConditionError('{0}: self.scheduler is None - scheduler must be assigned'.format(type(self).__name__))
            logger.debug('checking if {0} in previous execution step set'.format(dependency))
            try:
                return dependency in self.scheduler.execution_list[-1]
            except TypeError:
                return dependency == self.scheduler.execution_list[-1]
        super().__init__(dependency, func)


class AllHaveRun(Condition):
    """
    AllHaveRun

    Parameters:
        - *dependencies (Components): variable length
        - time_scale (TimeScale): the TimeScale used as basis for counting executions of dependencies. Defaults to TimeScale.TRIAL

    Satisfied when:
        - All dependencies have been executed at least 1 time within the scope of time_scale

    Notes:

    """
    def __init__(self, *dependencies, time_scale=TimeScale.TRIAL):
        def func(_none, *dependencies):
            if self.scheduler is None:
                raise ConditionError('{0}: self.scheduler is None - scheduler must be assigned'.format(type(self).__name__))
            if len(dependencies) == 0:
                dependencies = self.scheduler.nodes
            for d in dependencies:
                if self.scheduler.counts_total[time_scale][d] < 1:
                    return False
            return True
        super().__init__(None, func, *dependencies)


class WhenFinished(Condition):
    """
    WhenFinished

    Parameters:
        - dependency (Component):

    Satisfied when:
        - dependency has "finished" (i.e. its is_finished attribute is True)

    Notes:
        This is a dynamic condition.
        The is_finished concept varies among components, and is currently implemented in:
            `DDM`<DDM>

    """
    def __init__(self, dependency):
        def func(dependency):
            try:
                return dependency.is_finished
            except AttributeError as e:
                raise ConditionError('WhenFinished: Unsupported dependency type: {0}; ({1})'.format(type(dependency), e))

        super().__init__(dependency, func)


class WhenFinishedAny(Condition):
    """
    WhenFinishedAny

    Parameters:
        - *dependencies (Components): variable length

    Satisfied when:
        - any of the dependencies have "finished" (i.e. its is_finished attribute is True)

    Notes:
    This is a dynamic condition.
        This is a convenience class; WhenFinishedAny(A, B, C) is equivalent to Any(WhenFinished(A), WhenFinished(B), WhenFinished(C))
        If no dependencies are specified, the condition will default to checking all of its scheduler's Components.
        The is_finished concept varies among components, and is currently implemented in:
            `DDM`<DDM>

    """
    def __init__(self, *dependencies):
        def func(_none, *dependencies):
            if len(dependencies) == 0:
                dependencies = self.scheduler.nodes
            for d in dependencies:
                try:
                    if d.is_finished:
                        return True
                except AttributeError as e:
                    raise ConditionError('WhenFinishedAny: Unsupported dependency type: {0}; ({1})'.format(type(d), e))
            return False

        super().__init__(None, func, *dependencies)


class WhenFinishedAll(Condition):
    """
    WhenFinishedAll

    Parameters:
        - *dependencies (Components): variable length

    Satisfied when:
        - all of the dependencies have "finished" (i.e. its is_finished attribute is True)

    Notes:
        This is a dynamic condition.
        This is a convenience class; WhenFinishedAll(A, B, C) is equivalent to All(WhenFinished(A), WhenFinished(B), WhenFinished(C))
        If no dependencies are specified, the condition will default to checking all of its scheduler's Components.
        The is_finished concept varies among components, and is currently implemented in:
            `DDM`<DDM>

    """
    def __init__(self, *dependencies):
        def func(_none, *dependencies):
            if len(dependencies) == 0:
                dependencies = self.scheduler.nodes
            for d in dependencies:
                try:
                    if not d.is_finished:
                        return False
                except AttributeError as e:
                    raise ConditionError('WhenFinishedAll: Unsupported dependency type: {0}; ({1})'.format(type(d), e))
            return True

        super().__init__(None, func, *dependencies)
