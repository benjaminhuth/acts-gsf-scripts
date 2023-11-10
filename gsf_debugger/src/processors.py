import traceback
import re
import copy
import sys
from itertools import cycle

import matplotlib.pyplot as plt
import numpy as np

class BaseProcessor:
    def parse_line_base(self, line):
        try:
            self.parse_line(line)
        except:
            print("ERROR during parsing of line '{}'".format(line.rstrip()))
            print(traceback.format_exc())
            exit(1)


class AverageTrackPlotter(BaseProcessor):
    def __init__(self, view_drawers, annotate_steps=False):
        self.annotate_steps = annotate_steps

        self.fwd_steps = []
        self.bwd_steps = []

        self._current_steps = self.fwd_steps

        self.view_drawers = view_drawers

    def parse_line(self, line):
        if line.count("Do backward propagation") == 1:
            self._current_steps = self.bwd_steps

        elif line.count("at mean position") == 1:
            line = re.sub(r"^.*position", "", line)
            line = re.sub(r"with.*$", "", line)

            self._current_steps.append(
                np.array([float(item) for item in line.split()])
            )
            
    def name(self):
        return "Average track"
            
    def get_figure_axes(self):
        return  plt.subplots(1, len(self.view_drawers))
    
    def number_steps(self):
        return len(self.fwd_steps) + len(self.bwd_steps)

    def draw(self, fig, axes, step):
        for ax, drawer in zip(axes, self.view_drawers):
            ax = drawer.draw_detector(ax)

        if step > self.number_steps():
            return fig, axes
        
        fwd_to_plot = []
        bwd_to_plot = []
        
        if step < len(self.fwd_steps):
            fwd_to_plot = self.fwd_steps[:step]
        else:
            bwd_steps = step - len(self.fwd_steps)
            fwd_to_plot = self.fwd_steps
            bwd_to_plot = self.bwd_steps[:bwd_steps]

        positions = [fwd_to_plot, bwd_to_plot]
        names = ["forward", "backward"]

        for positions, direction in zip(positions, names):
            if len(positions) == 0:
                continue

            positions = np.vstack(positions)

            for ax, drawer in zip(axes, self.view_drawers):
                ax = drawer.plot(ax, positions, label=direction, marker='x', lw=1)

                if self.annotate_steps:
                    for i, p in enumerate(positions):
                        ax = drawer.annotate(ax, p, str(i))

                ax = drawer.set_title(ax)
                ax.legend()
                
        fig.tight_layout()
        return fig, axes


class ComponentsPlotter(BaseProcessor):
    def __init__(self, view_drawers):
        self.component_positions = []
        self.last_component = sys.maxsize
        self.current_direction = "forward"
        self.current_step = 0
        self.stages = []

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

    def name(self):
        return "Components"
            
    def get_figure_axes(self):
        return  plt.subplots(1, len(self.view_drawers))
    
    def number_steps(self):
        return sum([ len(s[3]) for s in self.stages])

    def draw_stage(self, fig, axes, stage):
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
        
        target_surface, direction, abs_step, component_positions = stage
        
        base_step = abs_step - len(component_positions)
        
        fig.suptitle(
            "Stepping {} towards {} ({} steps, starting from {})".format(
                direction,
                target_surface,
                len(component_positions),
                base_step,
            )
        )

        color_positions_x = {color: [] for color in colors}
        color_positions_z = {color: [] for color in colors}
        color_positions_y = {color: [] for color in colors}
        annotations = {color: [] for color in colors}

        for step, components in enumerate(component_positions):
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
                ax = drawer.plot(ax, color_positions, c=color, marker='x', lw=1)


    def draw(self, fig, axes, requested_step):        
        
        fig.suptitle("Nothing to draw\n")
        for ax, drawer in zip(axes, self.view_drawers):
            ax = drawer.draw_detector(ax)

        # print("req",requested_step)
        # print([s[2] for s in self.stages], flush=True)

        for s in self.stages:
            if s[2] >= requested_step:
                self.draw_stage(fig, axes, s)
                break

        fig.tight_layout()
        return fig, axes





class LogCollector(BaseProcessor):
    def __init__(self):
        self.loglines = []

        self._current_lines = []

    def parse_line(self, line):
        
        if line.count("at mean position") == 1:
            self.loglines.append(copy.deepcopy(self._current_lines))
            self._current_lines = []
            
        self._current_lines.append(line)


    def name(self):
        return "Log"

    def number_steps(self):
        return len(self.loglines)
        



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



class MomentumGraph(BaseProcessor):
    def __init__(self):
        self.gsf_forward_momentum_recorder = GsfMomentumRecorder()
        self.gsf_backward_momentum_recorder = GsfMomentumRecorder()
        self.gsf_current_momentum_recorder = self.gsf_forward_momentum_recorder

        self._flipped = False

    def parse_line(self, line):
        if line.count("Do backward propagation") == 1:
            self.gsf_current_momentum_recorder = self.gsf_backward_momentum_recorder

        self.gsf_current_momentum_recorder.parse_line(line)

    def name(self):
        return "Momentum"

    def get_figure_axes(self):
        return plt.subplots()

    def number_steps(self):
        return len(self.gsf_forward_momentum_recorder.gsf_momenta) + len(self.gsf_backward_momentum_recorder.gsf_momenta)

    def plot_components(self, ax, pathlengths, cmp_data, colors):
        for pl, (momenta, weights), color in zip(pathlengths, cmp_data, cycle(colors)):
            s = np.array(weights) * 0.9 + 0.1
            ax.scatter(
                pl * np.ones(len(momenta)),
                momenta,
                c=color,
                s=s,
                label="_nolegend_",
            )

    def plot(
        self,
        ax,
        name,
        gsf_pathlengths,
        gsf_momenta,
        gsf_cmp_data,
        color
    ):
        # colors = ['red', 'orangered', 'orange', 'gold', 'olive', 'forestgreen', 'lime', 'teal', 'cyan', 'blue', 'indigo', 'magenta', 'brown']
        # colors = ['forestgreen', 'lime', 'teal', 'indigo']
        colors = ["black"]

        # Shorten if necessary
        if len(gsf_momenta) == len(gsf_pathlengths) + 1:
            gsf_momenta = gsf_momenta[:-1]

        # Ensure size is equal...
        gsf_cmp_data = gsf_cmp_data[0 : len(gsf_momenta)]

        # Scatter components
        # self.plot_components(ax, gsf_pathlengths, gsf_cmp_data, colors)

        if len(gsf_pathlengths) > 0:
            ax.plot(
                gsf_pathlengths,
                gsf_momenta,
                c=color,
                label="GSF {}".format(name),
            )

        return ax


    def draw(self, fig, ax, requested_step):
        if not self._flipped:
            self.gsf_backward_momentum_recorder.gsf_pathlengths = -1*np.flip(np.array(self.gsf_backward_momentum_recorder.gsf_pathlengths))
            self._flipped = True

        all_pls = np.concatenate([self.gsf_forward_momentum_recorder.gsf_pathlengths, self.gsf_backward_momentum_recorder.gsf_pathlengths])


        # Forward
        ax = self.plot(
            ax,
            "forward",
            self.gsf_forward_momentum_recorder.gsf_pathlengths,
            self.gsf_forward_momentum_recorder.gsf_momenta,
            self.gsf_forward_momentum_recorder.gsf_cmp_data,
            "tab:blue",
        )

        # Backward
        ax = self.plot(
            ax,
            "backward",
            self.gsf_backward_momentum_recorder.gsf_pathlengths,
            self.gsf_backward_momentum_recorder.gsf_momenta,
            self.gsf_backward_momentum_recorder.gsf_cmp_data,
            "tab:orange",
        )

        ax.legend()
        ax.grid(which="both")

        ax.set_ylabel("momentum [GeV]")
        ax.set_xlabel("pathlength [mm]")

        if requested_step < len(all_pls):
            ax.vlines(x=all_pls[requested_step], ymin=ax.get_ylim()[0],
                      ymax=ax.get_ylim()[1], alpha=0.5,
                      color="tab:blue" if requested_step < len(self.gsf_forward_momentum_recorder.gsf_pathlengths) else "tab:orange")
