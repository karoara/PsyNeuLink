import numpy as np

from psyneulink.components.functions.function import Logistic
from psyneulink.components.mechanisms.processing.transfermechanism import TransferMechanism
from psyneulink.components.mechanisms.adaptive.learning.learningmechanism import LearningMechanism
from psyneulink.components.process import Process
from psyneulink.components.projections.pathway.mappingprojection import MappingProjection
from psyneulink.components.projections.modulatory.learningprojection import LearningProjection
from psyneulink.components.system import System
from psyneulink.globals.keywords import SOFT_CLAMP, EXECUTION, LEARNING, VALUE
from psyneulink.globals.preferences.componentpreferenceset import REPORT_OUTPUT_PREF, VERBOSE_PREF
from psyneulink.library.mechanisms.processing.objective.comparatormechanism import MSE

class TestBackPropExplicit:
    def test_single_layer(self):
        a = TransferMechanism(name="a")
        b = TransferMechanism(name="b")
        mapping_projection = MappingProjection(name="mapping_projection")
        p = Process(pathway=[a, mapping_projection, b])
        learning_projection = LearningProjection(receiver=mapping_projection,
                                                 name="learning_projection")
        learning_mechanism = LearningMechanism(default_variable=[[0.], [0.], [0.]],
                                               error_sources=[b],
                                               learning_signals=[learning_projection]
                                               )
        s = System(processes=[p])



