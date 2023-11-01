import traceback
import re

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
                ax = drawer.plot(ax, positions, label=direction)
                ax = drawer.scatter(ax, positions)

                if self.annotate_steps:
                    for i, p in enumerate(positions):
                        ax = drawer.annotate(ax, p, str(i))

                ax = drawer.set_title(ax)
                ax.legend()
                
        fig.tight_layout()
        return fig, axes


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

