import numpy as np
from psyneulink.components.mechanisms.processing.transfermechanism import TransferMechanism
from psyneulink.components.mechanisms.adaptive.learning.learningmechanism import LearningMechanism
from psyneulink.components.process import Process
from psyneulink.components.projections.pathway.mappingprojection import MappingProjection
from psyneulink.components.projections.modulatory.learningprojection import LearningProjection
from psyneulink.components.system import System
from psyneulink.globals.keywords import SOFT_CLAMP, EXECUTION, LEARNING, VALUE, SAMPLE
from psyneulink.globals.preferences.componentpreferenceset import REPORT_OUTPUT_PREF, VERBOSE_PREF
from psyneulink.library.mechanisms.processing.objective.comparatormechanism import MSE
from psyneulink.components.functions.function import BackPropagation
from psyneulink.library.mechanisms.processing.objective.comparatormechanism import ComparatorMechanism
class TestBackPropExplicit:
    def test_single_layer(self):
        # Create processing components:
        a = TransferMechanism(name="a")
        b = TransferMechanism(name="b")
        mapping_projection = MappingProjection(name="mapping_projection")
        p = Process(pathway=[a, mapping_projection, b])

        # Create learning components
        learning_function = BackPropagation(default_variable=[[0.], [0.], [0.]])
        learning_projection = LearningProjection(receiver=mapping_projection,
                                                 name="learning_projection")
        error_source = ComparatorMechanism(default_variable=[[0.], [0.]])
        learning_mechanism = LearningMechanism(default_variable=[[0.], [0.], [0.]],
                                               error_sources=[error_source],
                                               learning_signals=[learning_projection],
                                               function=learning_function
                                               )
        sample_projection = MappingProjection(sender=b, receiver=error_source.input_states[SAMPLE])
        s = System(processes=[p])

        s.show_graph(show_learning=True)

