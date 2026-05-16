# ============================================================
# Adaptive Lab Guardian — Fuzzy Decision Engine
# ============================================================

import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl


class AdaptiveGuardianFuzzy:
    """
    Adaptive Lab Guardian
    ----------------------------------------
    Inputs:
        - ART2  -> anomaly score
        - RBF   -> trend velocity
        - SOM   -> cluster id

    Outputs:
        - Scenario classification
        - Risk level
    """

    # ========================================================
    # Initialization
    # ========================================================

    def __init__(self):

        self._build_inputs()
        self._build_outputs()
        self._build_memberships()
        self._build_rules()
        self._build_system()

    # ========================================================
    # Inputs
    # ========================================================

    def _build_inputs(self):

        # ART2 → anomaly score [0 → 1]
        self.anomaly = ctrl.Antecedent(
            np.arange(0, 1.01, 0.01),
            'art_anomaly'
        )

        # RBF → trend velocity [-5 → +5]
        self.trend = ctrl.Antecedent(
            np.arange(-5, 5.1, 0.1),
            'rbf_trend'
        )

        # SOM → cluster id [0 → 3]
        # 0 = normal
        # 1 = crowded
        # 2 = chemical
        # 3 = security
        self.cluster = ctrl.Antecedent(
            np.arange(0, 3.1, 0.1),
            'som_cluster'
        )

    # ========================================================
    # Outputs
    # ========================================================

    def _build_outputs(self):

        # Scenario Output
        self.scenario = ctrl.Consequent(
            np.arange(0, 4.1, 0.1),
            'scenario'
        )

        # Risk Output
        self.risk = ctrl.Consequent(
            np.arange(0, 101, 1),
            'risk_level'
        )

        # Better defuzzification
        self.scenario.defuzzify_method = 'centroid'
        self.risk.defuzzify_method = 'centroid'

    # ========================================================
    # Membership Functions
    # ========================================================

    def _build_memberships(self):

        # ----------------------------------------------------
        # A) Anomaly Score
        # ----------------------------------------------------

        self.anomaly['low'] = fuzz.gaussmf(
            self.anomaly.universe,
            0.15,
            0.12
        )

        self.anomaly['medium'] = fuzz.gaussmf(
            self.anomaly.universe,
            0.50,
            0.15
        )

        self.anomaly['high'] = fuzz.gaussmf(
            self.anomaly.universe,
            0.85,
            0.12
        )

        # ----------------------------------------------------
        # B) Trend Velocity
        # ----------------------------------------------------

        self.trend['negative'] = fuzz.gaussmf(
            self.trend.universe,
            -3,
            1.2
        )

        self.trend['stable'] = fuzz.gaussmf(
            self.trend.universe,
            0,
            0.8
        )

        self.trend['positive'] = fuzz.gaussmf(
            self.trend.universe,
            3,
            1.2
        )

        # ----------------------------------------------------
        # C) SOM Cluster
        # ----------------------------------------------------

        self.cluster['normal'] = fuzz.gaussmf(
            self.cluster.universe,
            0,
            0.25
        )

        self.cluster['crowded'] = fuzz.gaussmf(
            self.cluster.universe,
            1,
            0.25
        )

        self.cluster['chemical'] = fuzz.gaussmf(
            self.cluster.universe,
            2,
            0.25
        )

        self.cluster['security'] = fuzz.gaussmf(
            self.cluster.universe,
            3,
            0.25
        )

        # ----------------------------------------------------
        # D) Scenario Output
        # ----------------------------------------------------

        self.scenario['routine'] = fuzz.trimf(
            self.scenario.universe,
            [0, 0, 1]
        )

        self.scenario['crowded'] = fuzz.trimf(
            self.scenario.universe,
            [0.8, 1.5, 2.2]
        )

        self.scenario['hazardous'] = fuzz.trimf(
            self.scenario.universe,
            [1.8, 2.5, 3]
        )

        self.scenario['breach'] = fuzz.trimf(
            self.scenario.universe,
            [3, 4, 4]
        )

        # ----------------------------------------------------
        # E) Risk Output
        # ----------------------------------------------------

        self.risk['safe'] = fuzz.trimf(
            self.risk.universe,
            [0, 0, 40]
        )

        self.risk['warning'] = fuzz.trimf(
            self.risk.universe,
            [25, 50, 75]
        )

        self.risk['critical'] = fuzz.trimf(
            self.risk.universe,
            [60, 100, 100]
        )

    # ========================================================
    # Rules
    # ========================================================

    def _build_rules(self):

        self.rules = []

        def add_rule(condition, scenario_term, risk_term):

            # Scenario Rule
            self.rules.append(
                ctrl.Rule(
                    condition,
                    self.scenario[scenario_term]
                )
            )

            # Risk Rule
            self.rules.append(
                ctrl.Rule(
                    condition,
                    self.risk[risk_term]
                )
            )

        # ====================================================
        # ROUTINE
        # ====================================================

        add_rule(
            self.anomaly['low']
            & self.trend['stable']
            & self.cluster['normal'],
            'routine',
            'safe'
        )

        add_rule(
            self.anomaly['low']
            & self.trend['negative']
            & self.cluster['normal'],
            'routine',
            'safe'
        )

        # ====================================================
        # CROWDED
        # ====================================================

        add_rule(
            self.cluster['crowded']
            & self.anomaly['low'],
            'crowded',
            'warning'
        )

        add_rule(
            self.cluster['crowded']
            & self.anomaly['medium'],
            'crowded',
            'warning'
        )

        add_rule(
            self.cluster['crowded']
            & self.trend['positive'],
            'hazardous',
            'critical'
        )

        # ====================================================
        # HAZARDOUS
        # ====================================================

        add_rule(
            self.anomaly['high']
            & self.trend['positive'],
            'hazardous',
            'critical'
        )

        add_rule(
            self.cluster['chemical']
            & self.anomaly['medium'],
            'hazardous',
            'critical'
        )

        add_rule(
            self.cluster['chemical']
            & self.trend['positive'],
            'hazardous',
            'critical'
        )

        add_rule(
            self.cluster['chemical']
            & self.anomaly['high'],
            'hazardous',
            'critical'
        )

        # ====================================================
        # BREACH
        # ====================================================

        add_rule(
            self.cluster['security']
            & self.anomaly['high'],
            'breach',
            'critical'
        )

        add_rule(
            self.cluster['security']
            & self.trend['positive'],
            'breach',
            'critical'
        )

        add_rule(
            self.cluster['security']
            & self.anomaly['medium']
            & self.trend['positive'],
            'breach',
            'critical'
        )

        # ====================================================
        # ADAPTIVE LOGIC
        # ====================================================

        add_rule(
            self.anomaly['medium']
            & self.trend['positive'],
            'hazardous',
            'critical'
        )

        add_rule(
            self.anomaly['medium']
            & self.trend['negative'],
            'crowded',
            'warning'
        )

        add_rule(
            self.anomaly['high']
            & self.trend['negative'],
            'hazardous',
            'warning'
        )

        # ====================================================
        # EDGE CASES
        # ====================================================

        add_rule(
            self.anomaly['low']
            & self.trend['positive']
            & self.cluster['chemical'],
            'hazardous',
            'critical'
        )

        add_rule(
            self.anomaly['low']
            & self.trend['stable']
            & self.cluster['crowded'],
            'crowded',
            'warning'
        )

        add_rule(
            self.anomaly['low']
            & self.trend['stable']
            & self.cluster['chemical'],
            'hazardous',
            'warning'
        )

        add_rule(
            self.anomaly['low']
            & self.trend['stable']
            & self.cluster['security'],
            'breach',
            'warning'
        )

    # ========================================================
    # Build System
    # ========================================================

    def _build_system(self):

        self.control_system = ctrl.ControlSystem(self.rules)

    # ========================================================
    # Prediction
    # ========================================================

    def predict(self, anomaly_score, trend_velocity, cluster_id):

        sim = ctrl.ControlSystemSimulation(self.control_system)

        # Inputs
        sim.input['art_anomaly'] = anomaly_score
        sim.input['rbf_trend'] = trend_velocity
        sim.input['som_cluster'] = cluster_id

        # Compute
        sim.compute()

        # Raw outputs
        scenario_value = sim.output.get('scenario', 0)
        risk_value = sim.output.get('risk_level', 0)

        # ----------------------------------------------------
        # Scenario Classification
        # ----------------------------------------------------

        if scenario_value < 1:
            scenario_label = "Routine"

        elif scenario_value < 2:
            scenario_label = "Crowded"

        elif scenario_value < 3:
            scenario_label = "Hazardous"

        else:
            scenario_label = "Breach"

        # ----------------------------------------------------
        # Risk Classification
        # ----------------------------------------------------

        if risk_value < 40:
            risk_label = "Safe"

        elif risk_value < 70:
            risk_label = "Warning"

        else:
            risk_label = "Critical"

        # ----------------------------------------------------
        # Return Results
        # ----------------------------------------------------

        return {

            "scenario": scenario_label,
            "scenario_score": round(scenario_value, 2),

            "risk": risk_label,
            "risk_score": round(risk_value, 2)
        }

    # ========================================================
    # Visualization Helpers
    # ========================================================

    def visualize_inputs(self):

        self.anomaly.view()
        self.trend.view()
        self.cluster.view()

    def visualize_outputs(self):

        self.scenario.view()
        self.risk.view()