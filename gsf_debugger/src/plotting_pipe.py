#!/usr/bin/env python3
import pathlib
import sys
import os
import re
import datetime
import copy
import traceback
import subprocess
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
import pandas as pd
import numpy as np
from itertools import cycle
import argparse

######################
# ViewDrawer classes #
######################



#####################
# Processor classes #
#####################


class BaseProcessor:
    def parse_line_base(self, line):
        try:
            self.parse_line(line)
        except:
            print("ERROR during parsing of line '{}'".format(line.rstrip()))
            print(traceback.format_exc())
            exit(1)


class TrueParticleRecorder(BaseProcessor):
    def __init__(self):
        self.found_one_electron = False
        self.parse_momenta_and_pathlengths = False
        self.momenta = []
        self.pathlengths = []
        self.true_hits = []

    def parse(self, line):
        if line.count("FinalParticle") == 1 and not line.count("End") == 1:
            split = line.split()

            # Make sure we have an electron
            if int(split[-3]) != 11:
                return True

            if self.found_one_electron:
                print(
                    "Warning: found more than one electrons, expect exactely one electron"
                )
                return True

            self.found_one_electron = True
            self.parse_momenta_and_pathlengths = True
            return True

        elif line.count("End FinalParticle") == 1:
            self.parse_momenta_and_pathlengths = False
            return True

        elif self.parse_momenta_and_pathlengths:
            s = line.split()

            self.momenta.append(float(s[-1]))
            self.pathlengths.append(float(s[-3]))

            pos = np.array([float(s[-7]), float(s[-6]), float(s[-5])])
            self.true_hits.append(pos)
            return True

        # If nothing got parsed, return false
        return False


class CovarianceRecorder(BaseProcessor):
    def __init__(self, enabler="Gsf step", disabler="KalmanFitter step"):
        self.enabler = enabler
        self.disabler = disabler
        self.enabled = False

        self.read = -1
        self.current_cov = None

        self.covs = []

    def parse_line(self, line):
        if line.count(self.enabler) == 1:
            self.enabled = True
            return

        if line.count(self.disabler) == 1:
            self.enabled = False
            return

        if self.enabled == False:
            return

        if line.count("Predicted covariance") == 1:
            self.read = 0
            self.current_cov = np.zeros((6, 6))
            return

        if self.read == 6:
            self.covs.append(self.current_cov)
            self.current_cov = None
            self.read = -1
            return

        if self.read >= 0:
            self.current_cov[self.read, :] = np.array([float(s) for s in line.split()])
            self.read += 1
            return


class CovarianceProcessor(BaseProcessor):
    def __init__(self):
        self.gsf_recorder = CovarianceRecorder(
            enabler="Gsf step", disabler="KalmanFitter step"
        )
        self.kf_recorder = CovarianceRecorder(
            enabler="KalmanFitter step", disabler="Gsf step"
        )

    def parse_line(self, line):
        self.gsf_recorder.parse_line(line)
        self.kf_recorder.parse_line(line)

    def process_data(self):
        plt.plot([c[4, 4] for c in self.gsf_recorder.covs], label="gsf")
        plt.plot([c[4, 4] for c in self.kf_recorder.covs], label="kf")
        plt.legend()
        plt.title("QoP variance")


class GsfMomentumRecorder:
    def __init__(self):
        # Global state
        self.gsf_started_backwards = False
        self.gsf_accumulated_pathlength = 0
        self.printed_qop_warning = False

        # Recordings
        self.gsf_momenta = []
        self.gsf_cmp_data = []
        self.gsf_pathlengths = []

        # Current step
        self.gsf_current_step_cmp_momenta = []
        self.gsf_current_step_cmp_weights = []
        self.gsf_have_momentum = False
        self.gsf_last_step_number = None

    def parse_line(self, line):
        if line.count("Gsf step") == 1:
            # Last step appears twice in log, prevent this
            if int(line.split()[5]) == self.gsf_last_step_number:
                return

            self.gsf_last_step_number = int(line.split()[5])

            # Update component states if not in first step
            if len(self.gsf_current_step_cmp_momenta) > 0:
                self.gsf_cmp_data.append(
                    (
                        self.gsf_current_step_cmp_momenta,
                        self.gsf_current_step_cmp_weights,
                    )
                )
                self.gsf_current_step_cmp_momenta = []
                self.gsf_current_step_cmp_weights = []

            # Save momentum
            assert len(self.gsf_momenta) == len(self.gsf_pathlengths)
            self.gsf_momenta.append(float(line.split()[-4]))
            self.gsf_have_momentum = True

        elif re.match(r"^.*#[0-9]+\spos", line) and not self.gsf_started_backwards:
            line = line.replace(",", "")
            qop = float(line.split()[-3])
            if abs(1.0 / qop) < 1:
                p = qop
                if not self.printed_qop_warning:
                    print("WARNING: assume qop -> p because |1/qop| < 1")
                    self.printed_qop_warning = True
            p = abs(1 / qop)
            w = float(line.split()[-7])
            self.gsf_current_step_cmp_momenta.append(p)
            self.gsf_current_step_cmp_weights.append(w)

        elif line.count("Step with size") == 1:
            self.gsf_accumulated_pathlength += float(line.split()[-2])

            if self.gsf_have_momentum:
                self.gsf_have_momentum = False
                self.gsf_pathlengths.append(self.gsf_accumulated_pathlength)
                assert len(self.gsf_pathlengths) == len(self.gsf_momenta)


class KalmanMomentumRecorder:
    def __init__(self):
        self.kalman_momenta = []
        self.kalman_pathlengths = []
        self.kalman_momenta_backward = []
        self.kalman_pathlengths_backward = []

        self.momentum_list = self.kalman_momenta
        self.path_list = self.kalman_pathlengths

        # state
        self.kalman_current_pathlength = 0
        self.kalman_have_momentum = False
        self.kalman_started_smoothing = False

    def parse_line(self, line):
        if (
            line.count("KalmanFitter step at") == 1
            and not self.kalman_started_smoothing
        ):
            assert len(self.path_list) == len(self.momentum_list)
            self.momentum_list.append(float(line.split()[-1]))
            self.kalman_have_momentum = True

        if line.count("Reverse navigation direction") == 1:
            self.kalman_current_pathlength = 0
            self.momentum_list = self.kalman_momenta_backward
            self.momentum_list.append(self.kalman_momenta[-1])
            self.path_list = self.kalman_pathlengths_backward

        elif line.count("Step with size") == 1:
            if self.kalman_have_momentum:
                self.path_list.append(self.kalman_current_pathlength)
                assert len(self.path_list) == len(self.momentum_list)
                self.kalman_have_momentum = False

            self.kalman_current_pathlength += float(line.split()[-2])

        elif line.count("Finalize/run smoothing") == 1:
            self.kalman_started_smoothing = True


class MomentumGraph(BaseProcessor):
    def __init__(self, use_weights_as_opacity, annotate_steps):
        self.use_weights_as_opacity = use_weights_as_opacity
        self.annotate_steps = annotate_steps
        self.disabled = False

        self.true_particle_recorder = TrueParticleRecorder()

        self.gsf_forward_momentum_recorder = GsfMomentumRecorder()
        self.gsf_backward_momentum_recorder = GsfMomentumRecorder()
        self.gsf_last_part_momentum_recorder = GsfMomentumRecorder()
        self.gsf_current_momentum_recorder = self.gsf_forward_momentum_recorder
        self.in_gsf_mode = False

        self.kalman_momentum_recorder = KalmanMomentumRecorder()
        self.in_kalman_mode = False

    def parse_line(self, line):
        if self.disabled:
            return

        if self.true_particle_recorder.parse(line):
            return

        elif line.count("KalmanFitter step at") == 1:
            self.in_kalman_mode = True
            self.in_gsf_mode = False

        elif line.count("Gsf step") == 1:
            self.in_kalman_mode = False
            self.in_gsf_mode = True

        elif line.count("Do backward propagation") == 1:
            self.gsf_current_momentum_recorder = self.gsf_backward_momentum_recorder

        elif line.count("Do propagation back to reference surface") == 1:
            self.gsf_current_momentum_recorder = self.gsf_last_part_momentum_recorder
            return

        if self.in_gsf_mode:
            self.gsf_current_momentum_recorder.parse_line(line)
            return

        if self.in_kalman_mode:
            self.kalman_momentum_recorder.parse_line(line)
            return

    def plot_components(self, ax, pathlengths, cmp_data, colors):
        for pl, (momenta, weights), color in zip(pathlengths, cmp_data, cycle(colors)):
            if self.use_weights_as_opacity:
                weights = np.array(weights) * 0.9 + 0.1

                r, g, b, a = to_rgba(color)

                color = np.zeros((len(momenta), 4))
                color[:, 0] = r
                color[:, 1] = g
                color[:, 2] = b
                color[:, 3] = weights

            ax.scatter(
                pl * np.ones(len(momenta)),
                momenta,
                c=color,
                s=8,
                label="_nolegend_",
            )

    def plot(
        self,
        ax,
        name,
        gsf_pathlengths,
        gsf_momenta,
        gsf_cmp_data,
        kalman_pathlengths,
        kalman_momenta,
    ):
        # colors = ['red', 'orangered', 'orange', 'gold', 'olive', 'forestgreen', 'lime', 'teal', 'cyan', 'blue', 'indigo', 'magenta', 'brown']
        # colors = ['forestgreen', 'lime', 'teal', 'indigo']
        colors = ["black"]

        # Shorten if necessary
        if len(gsf_momenta) == len(gsf_pathlengths) + 1:
            gsf_momenta = gsf_momenta[:-1]

        # Ensure size is equal...
        gsf_cmp_data = gsf_cmp_data[0 : len(gsf_momenta)]

        # Shorten if necessary
        if len(kalman_momenta) == len(kalman_pathlengths) + 1:
            kalman_momenta = kalman_momenta[:-1]

        # Annotate step numbers
        if self.annotate_steps:
            for i, xy in enumerate(zip(gsf_pathlengths, gsf_momenta)):
                ax.annotate(str(i), xy)

        # Scatter components
        self.plot_components(ax, gsf_pathlengths, gsf_cmp_data, colors)

        if len(gsf_pathlengths) > 0:
            ax.plot(
                gsf_pathlengths,
                gsf_momenta,
                linewidth=3,
                c="blue",
                label="GSF {}".format(name),
            )
            ax.scatter(
                gsf_pathlengths,
                gsf_momenta,
                linewidth=3,
                c="blue",
                label="GSF {}".format(name),
            )

        if len(kalman_pathlengths) > 0:
            ax.plot(
                kalman_pathlengths,
                kalman_momenta,
                linewidth=3,
                c="orange",
                label="Kalman {}".format(name),
            )
            ax.scatter(
                kalman_pathlengths,
                kalman_momenta,
                linewidth=3,
                c="orange",
                label="Kalman {}".format(name),
            )

        return ax

    def process_data(self):
        fig, ax = plt.subplots(1, 1)

        print(
            "self.gsf_forward_momentum_recorder.gsf_pathlengths",
            len(self.gsf_forward_momentum_recorder.gsf_pathlengths),
        )
        print(
            "self.kalman_momentum_recorder.kalman_pathlengths",
            len(self.kalman_momentum_recorder.kalman_pathlengths),
        )

        # Forward
        ax = self.plot(
            ax,
            "forward",
            self.gsf_forward_momentum_recorder.gsf_pathlengths,
            self.gsf_forward_momentum_recorder.gsf_momenta,
            self.gsf_forward_momentum_recorder.gsf_cmp_data,
            self.kalman_momentum_recorder.kalman_pathlengths,
            self.kalman_momentum_recorder.kalman_momenta,
        )

        # Augment last pathlengths (shoudl do nothing if empty)
        last_pathlengths = [
            pl + self.gsf_backward_momentum_recorder.gsf_pathlengths[-1]
            for pl in self.gsf_last_part_momentum_recorder.gsf_pathlengths
        ]

        # Shorten if necessary
        if (
            len(self.gsf_last_part_momentum_recorder.gsf_momenta)
            == len(last_pathlengths) + 1
        ):
            self.gsf_last_part_momentum_recorder.gsf_momenta = (
                self.gsf_last_part_momentum_recorder.gsf_momenta[:-1]
            )

        # Add potential last part (should do nothing if lists are empty)
        self.gsf_backward_momentum_recorder.gsf_pathlengths += last_pathlengths
        self.gsf_backward_momentum_recorder.gsf_momenta += (
            self.gsf_last_part_momentum_recorder.gsf_momenta
        )
        self.gsf_backward_momentum_recorder.gsf_cmp_data += (
            self.gsf_last_part_momentum_recorder.gsf_cmp_data
        )

        # Backward
        # plot (plot from negative to 0 for simplicity)
        def mirror_pathlengths(pls):
            pls += abs(min(pls))
            pls *= -1
            return pls

        gsf_bwd_pls = mirror_pathlengths(
            np.array(self.gsf_backward_momentum_recorder.gsf_pathlengths)
        )

        if len(self.kalman_momentum_recorder.kalman_pathlengths_backward) > 0:
            kalman_bwd_pls = mirror_pathlengths(
                np.array(self.kalman_momentum_recorder.kalman_pathlengths_backward)
            )
        else:
            kalman_bwd_pls = []

        ax = self.plot(
            ax,
            "backward",
            gsf_bwd_pls,
            self.gsf_backward_momentum_recorder.gsf_momenta,
            self.gsf_backward_momentum_recorder.gsf_cmp_data,
            kalman_bwd_pls,
            self.kalman_momentum_recorder.kalman_momenta_backward,
        )

        # True if available
        if len(self.true_particle_recorder.pathlengths) > 0:
            ax.plot(
                self.true_particle_recorder.pathlengths,
                self.true_particle_recorder.momenta,
                linewidth=3,
                c="darkred",
                label="True momentum",
            )

        # General plot settings
        ax.legend()
        ax.grid(which="both")

        ax.set_ylabel("momentum [GeV]")
        ax.set_xlabel("pathlength [mm]")

        # In this case do log scale so we can view everything well
        if ax.get_ylim()[1] > 20:  # and not ax.get_ylim()[0] < 0:
            ax.set_yscale("log")

        ax.axvline(x=0, ymin=1.0e-3, ymax=ax.get_ylim()[1], c="black")

        fig.suptitle("Momentum Graph")


class KalmanPlotter(BaseProcessor):
    def __init__(self, view_drawers):
        self.view_drawers = view_drawers

        self.global_positions_forward = []
        self.global_positions_backward = []

        self.current_position_list = self.global_positions_forward
        self.started_smoothing = False

        self.true_particle_recorder = TrueParticleRecorder()

    def parse_line(self, line):
        if self.true_particle_recorder.parse(line):
            return

        if self.started_smoothing:
            return

        if line.count("Reverse navigation direction") == 1:
            self.current_position_list = self.global_positions_backward

        if line.count("KalmanFitter step at") == 1:
            split = line.split()

            self.current_position_list.append(
                np.array([float(split[-9]), float(split[-8]), float(split[-7])])
            )

        elif line.count("Finalize/run smoothing") == 1:
            self.started_smoothing = True

    def process_data(self):
        if len(self.global_positions_forward) == 0:
            print("KalmanPlotter has no data to process. Return.")
            return

        fig, axes = plt.subplots(1, len(self.view_drawers))

        fig.suptitle("Kalman Fitter Track")

        pos_lists = [self.global_positions_forward]
        names = ["forward"]

        if len(self.global_positions_backward) > 0:
            pos_lists.append(self.global_positions_backward)
            names.append("backward")

        for pos_list, name in zip(pos_lists, names):
            global_pos = np.vstack(pos_list)

            for ax, drawer in zip(axes, self.view_drawers):
                ax = drawer.draw_detector(ax)
                ax = drawer.plot(ax, global_pos, label=name)
                ax = drawer.scatter(ax, global_pos)
                ax = drawer.set_title(ax)

                try:
                    true_hits = np.vstack(self.true_particle_recorder.true_hits)
                    ax = ax.scatter(true_hits, marker="*")
                except:
                    print("INFO: Could not read true hits")

                ax.legend()


class AverageTrackPlotter(BaseProcessor):
    def __init__(self, view_drawers, annotate_steps=False):
        self.annotate_steps = annotate_steps

        self.fwd_positions = []
        self.bwd_positions = []
        self.last_positions = []

        self.current_positions = self.fwd_positions

        self.true_particle_recorder = TrueParticleRecorder()

        self.view_drawers = view_drawers

    def parse_line(self, line):
        if self.true_particle_recorder.parse(line):
            return

        elif line.count("Do backward propagation") == 1:
            self.current_positions = self.bwd_positions

        elif line.count("Do propagation back to reference surface") == 1:
            self.current_positions = self.last_positions

        elif line.count("at mean position") == 1:
            line = re.sub(r"^.*position", "", line)
            line = re.sub(r"with.*$", "", line)

            self.current_positions.append(
                np.array([float(item) for item in line.split()])
            )

    def process_data(self):
        fig, axes = plt.subplots(1, len(self.view_drawers))

        fig.suptitle(
            "GSF average track (forward: {} steps, backward: {} steps, last: {} steps)".format(
                len(self.fwd_positions),
                len(self.bwd_positions),
                len(self.last_positions),
            )
        )

        positions = [
            self.fwd_positions,
            self.bwd_positions,
            self.last_positions,
        ]
        names = ["forward", "backward", "last"]

        for positions, direction in zip(positions, names):
            if len(positions) == 0:
                return

            positions = np.vstack(positions)

            for ax, drawer in zip(axes, self.view_drawers):
                ax = drawer.draw_detector(ax)
                ax = drawer.plot(ax, positions, label=direction)
                ax = drawer.scatter(ax, positions)

                if self.annotate_steps:
                    for i, p in enumerate(positions):
                        ax = drawer.annotate(ax, p, str(i))

                try:
                    true_hits = np.vstack(self.true_particle_recorder.true_hits)
                    ax = drawer.scatter(ax, true_hits, marker="*")
                except:
                    print("Warning: Could not print true hits")

                ax = drawer.set_title(ax)
                ax.legend()


class ComponentsPlotter(BaseProcessor):
    def __init__(self, view_drawers, pick_step=-1):
        self.component_positions = []
        self.last_component = sys.maxsize
        self.current_direction = "forward"
        self.current_step = 0
        self.stages = []
        self.pick_step = pick_step

        self.view_drawers = view_drawers

    def parse_line(self, line):
        if line.count("Step is at surface") == 1:
            surface_name = line[line.find("vol") :]

            self.stages.append(
                (
                    surface_name,
                    self.current_direction,
                    self.current_step,
                    copy.deepcopy(self.component_positions),
                )
            )
            self.component_positions = []
            self.last_component = sys.maxsize

        elif line.count("Do backward propagation") == 1:
            self.current_direction = "backward"
            self.current_step = 0

        elif re.match(r"^.*#[0-9]+\spos", line):
            line = line.replace(",", "")
            splits = line.split()

            current_cmp = int(splits[3][1:])

            pos = np.array([float(part) for part in splits[5:8]])

            if current_cmp < self.last_component:
                self.component_positions.append([pos])
                self.current_step += 1
            else:
                self.component_positions[-1].append(pos)

            self.last_component = current_cmp

    def process_data(self):
        colors = [
            "red",
            "orangered",
            "orange",
            "gold",
            "olive",
            "forestgreen",
            "lime",
            "teal",
            "cyan",
            "blue",
            "indigo",
            "magenta",
            "brown",
        ]

        for (
            target_surface,
            direction,
            abs_step,
            component_positions,
        ) in self.stages:
            if self.pick_step != -1 and abs_step != self.pick_step:
                continue

            fig, axes = plt.subplots(1, len(self.view_drawers))

            base_step = abs_step - len(component_positions)
            fig.suptitle(
                "Stepping {} towards {} ({} steps, starting from {})".format(
                    direction,
                    target_surface,
                    len(component_positions),
                    base_step,
                )
            )

            for ax, drawer in zip(axes, self.view_drawers):
                ax = drawer.draw_detector(ax)

            color_positions_x = {color: [] for color in colors}
            color_positions_z = {color: [] for color in colors}
            color_positions_y = {color: [] for color in colors}
            annotations = {color: [] for color in colors}

            for step, components in enumerate(component_positions):
                if step > 300:
                    break

                positions = np.vstack(components)

                for i, (cmp_pos, color) in enumerate(zip(positions, cycle(colors))):
                    color_positions_x[color].append(cmp_pos[0])
                    color_positions_z[color].append(cmp_pos[2])
                    color_positions_y[color].append(cmp_pos[1])

                    annotations[color].append("{}-{}".format(step, i))

            for color in colors:
                color_positions = np.array(
                    [
                        color_positions_x[color],
                        color_positions_y[color],
                        color_positions_z[color],
                    ]
                ).T

                for ax, drawer in zip(axes, self.view_drawers):
                    ax = drawer.scatter(ax, color_positions, c=color)

            plt.show()


class GsfKalmanPlotCombined(BaseProcessor):
    def __init__(self, view_drawers):
        self.view_drawers = view_drawers

        self.gsf_parser = AverageTrackPlotter(view_drawers)
        self.kalman_parser = KalmanPlotter(view_drawers)

    def parse_line(self, line):
        self.gsf_parser.parse_line(line)
        self.kalman_parser.parse_line(line)

    def plot(self, positions, names, title):
        fig, axes = plt.subplots(1, len(self.view_drawers))

        fig.suptitle(title)

        for pos, name in zip(positions, names):
            for ax, drawer in zip(axes, self.view_drawers):
                ax = drawer.draw_detector(ax)
                ax = drawer.plot(ax, pos, label=name)
                ax = drawer.scatter(ax, pos)

                ax = drawer.set_title(ax)
                ax.legend()

    def process_data(self):
        self.plot(
            [
                np.vstack(self.kalman_parser.global_positions_forward),
                np.vstack(self.gsf_parser.fwd_positions),
            ],
            ["kalman-forward", "gsf-forward"],
            "Forward pass comparison",
        )

        self.plot(
            [
                np.vstack(self.kalman_parser.global_positions_backward),
                np.vstack(self.gsf_parser.bwd_positions),
            ],
            ["kalman-backward", "gsf-backward"],
            "Backward pass comparison",
        )




ldmx_surfaces = np.array(
    [
        [
            -613,
            -20.955,
            0,
        ],
        [
            -607,
            -20.955,
            0,
        ],
        [
            -513,
            -14.643,
            0,
        ],
        [
            -507,
            -14.643,
            0,
        ],
        [
            -413,
            -9.461,
            0,
        ],
        [
            -407,
            -9.461,
            0,
        ],
        [
            -313,
            -5.407,
            0,
        ],
        [
            -307,
            -5.407,
            0,
        ],
        [
            -213,
            -2.481,
            0,
        ],
        [
            -207,
            -2.481,
            0,
        ],
        [
            -113,
            -0.681,
            0,
        ],
        [
            -107,
            -0.681,
            0,
        ],
        [
            -13,
            -0.006,
            0,
        ],
        [
            -7,
            -0.006,
            0,
        ],
        [
            -613,
            -20.955,
            0,
        ],
        [
            -607,
            -20.955,
            0,
        ],
        [
            -513,
            -14.643,
            0,
        ],
        [
            -507,
            -14.643,
            0,
        ],
        [
            -413,
            -9.461,
            0,
        ],
        [
            -407,
            -9.461,
            0,
        ],
        [
            -313,
            -5.407,
            0,
        ],
        [
            -307,
            -5.407,
            0,
        ],
        [
            -213,
            -2.481,
            0,
        ],
        [
            -207,
            -2.481,
            0,
        ],
        [
            -113,
            -0.681,
            0,
        ],
        [
            -107,
            -0.681,
            0,
        ],
        [
            -13,
            -0.006,
            0,
        ],
        [
            -7,
            -0.006,
            0,
        ],
    ]
)


class LdmxXYDrawer(XYDrawer):
    def __init__(self):
        super().__init__()
        self.x = ldmx_surfaces[:, 0]
        self.y = ldmx_surfaces[:, 1]
        self.half_length = 19.170

    def draw_detector_impl(self, ax):
        # ax.scatter([0,], [0,], color="grey")

        for x, y in zip(self.x, self.y):
            ax.plot(
                3 * [x],
                [y + self.half_length, y, y - self.half_length],
                color="grey",
            )

        return ax


class LdmxXZDrawer(XZDrawer):
    def __init__(self):
        super().__init__()
        self.x = ldmx_surfaces[:, 0]
        self.z = ldmx_surfaces[:, 2]
        self.half_length = 49.165

    def draw_detector_impl(self, ax):
        # ax.scatter([0,], [0,], color="grey")

        for x, z in zip(self.x, self.z):
            ax.plot(
                3 * [x],
                [z + self.half_length, z, z - self.half_length],
                color="grey",
            )

        return ax


#####################
# Manage processors #
#####################

if __name__ == "__main__":
    input_processors = {
        "avgplot": lambda drawers: AverageTrackPlotter(drawers),
        "avgplot-steps": lambda drawers: AverageTrackPlotter(
            drawers, annotate_steps=True
        ),
        "cmpplot": lambda drawers: ComponentsPlotter(drawers),
        "kalmanplot": lambda drawers: KalmanPlotter(drawers),
        "momentumgraph": lambda drawers: MomentumGraph(False, annotate_steps=True),
        "momentumgraph-opacity": lambda drawers: MomentumGraph(
            True, annotate_steps=True
        ),
        "gsfkalmanplot": lambda drawers: GsfKalmanPlotCombined(drawers),
        "qopvar": lambda drawers: CovarianceProcessor(),
    }

    parser = argparse.ArgumentParser(description="Run GSF Telescope")
    parser.add_argument(
        "processors",
        choices=list(input_processors.keys()) + ["all"],
        help="which processors geometry to use",
        nargs="+",
    )
    parser.add_argument("--geometry-csv", default="detectors.csv")
    parser.add_argument(
        "--geometry-type",
        choices=["csv", "csv-telescope", "ldmx"],
        default="csv",
    )
    args = vars(parser.parse_args())

    if args["geometry_type"] == "csv":
        detector_file = pathlib.Path(args["geometry_csv"])
        assert detector_file.exists()
        drawers = [CsvZRDrawer(detector_file), CsvXYDrawer(detector_file)]
    elif args["geometry_type"] == "csv-telescope":
        detector_file = pathlib.Path(args["geometry_csv"])
        assert detector_file.exists()
        drawers = [
            CsvXYDrawer(detector_file, assume_telescope=True),
            CsvXZDrawer(detector_file, assume_telescope=True),
        ]
    elif args["geometry_type"] == "ldmx":
        drawers = [LdmxXYDrawer(), LdmxXZDrawer()]

    print("Config:", args)

    selected_processors = []

    if "all" in args["processors"]:
        for key in input_processors.keys():
            selected_processors.append(input_processors[key](drawers))
    else:
        for key in args["processors"]:
            selected_processors.append(input_processors[key](drawers))

    if len(selected_processors) == 0:
        print(
            "No valid component selected. Chose one of {}".format(
                input_processors.keys()
            )
        )
        exit(1)

    ###########################
    # Run processors on input #
    ###########################

    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    lines = 0

    # with open("outfile-{}.txt".format(timestamp), 'w') as stdout_file:
    # for line in sys.stdin.readlines():
    # stdout_file.write(line)
    # lines += 1
    # for name in selected_processors:
    # input_processors[name].parse_line(line)

    for line in sys.stdin.readlines():
        lines += 1
        for processor in selected_processors:
            processor.parse_line_base(line)

    print("Parsed {} lines, process input now...".format(lines))

    for processor in selected_processors:
        processor.process_data()

    plt.show()
