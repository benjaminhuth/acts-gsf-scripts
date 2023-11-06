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
                ax = drawer.draw_detector(ax)
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
        
