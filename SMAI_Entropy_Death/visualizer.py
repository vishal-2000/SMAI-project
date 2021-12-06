'''
Optimizer Performance Visualizer

Key points:
- Uses PyTorch, Matplotlib
- Can be used to visualize optimizers written in PyTorch (both in-built or custom)

Author: Vishal Reddy Mandadi 
Team: Entropy Death 
Course: SMAI

References: 
1. https://github.com/Jaewan-Yun/optimizer-visualization,
2. https://github.com/3springs/viz_torch_optim
3. https://en.wikipedia.org/wiki/Rosenbrock_function

'''

from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.optim import Adam, SGD, Adagrad, RMSprop
from tqdm import tqdm

from ADAM import ADAM

class Visualizer:
    def __init__(self):
        pass

    def polynomial_cost(self, function_name='single-saddle1', x_tuple = None):
        '''Polynomial costs
        Parameters
        -------------------
        x_tuple : (x1, x2)
                    Two element tuple of floats representing x and y coordinates in 3D world
        function_name:  1. 'two-minima-eq': Has two equal minimas and one ridge
                                - z = ((x1+x2-2)**2)*((x1+x2+2)**2)
                                - minima: x+y=2 | x+y=-2
                                - maxima: x+y=0
                        2. 'two-minima-ueq': Has two unequal minimas (1 global) and one ridge
                                - z = ((x1+x2-4)**2)*((x1+x2+2)**2 - 0.5) 
                                - minima: x+y=4 | x+y=-2 (needs verification)
                                - maxima: x+y=0
                        3. 'single-saddle1': Contains saddle point (curve rises on one side, falls on the other)
                                - z = (x1-c1)**2 - (x2-c2)**2  
                                - saddle at (c1, c2)
                        4. 'single-saddle2': Contains saddle point (curve rises on one side, falls on the other)
                                - z = (x-c1)*(y-c2) 
                                - saddle at (0, 0)
                        5. 'rosenbrock': Standard rosenbrock function used to visualize optimizers in general
                                - z = (1 - x1) ** 2 + 100 * (x2 - x1 ** 2) ** 2
        Returns
        ---------------------
        float
            - Function evaluated at x_tuple
        Description
        ---------
        This function creates the contours in 3D space using 3 variables (x, y, z). Loosely based on 
        Taylor's expansions (any curve can be expressed as a polynomial). 
        polynomial_cost = z = f(x, y) will then be used for visualizing the optimizers
        '''
        # if x_tuple == None:
        #     print('Invalid initial values of x1 and x2, please enter a valid tuple (x1, x2)')
        #     exit()
        x1, x2 = x_tuple
        if function_name=='two-minima-eq':
            return ((x1+x2-2)**2)*((x1+x2+2)**2)
        elif function_name=='two-minima-ueq':
            return ((x1+x2-2)**2)*((x1+x2+2)**2 - 0.5)
        elif function_name=='single-saddle1':
            return (x1)**2 - (x2)**2
        elif function_name=='single-saddle2':
            return x1*x2
        elif function_name=='rosenbrock':
            return (1 - x1) ** 2 + 100 * (x2 - x1 ** 2) ** 2

    def two_minima_eq(self, x_tuple):
        x1, x2 = x_tuple
        return ((x1+x2-2)**2)*((x1+x2+2)**2)
    
    def two_minima_ueq(self, x_tuple):
        x1, x2 = x_tuple
        return ((x1+x2-2)**2)*((x1+x2+2)**2 - 0.5)
    
    def single_saddle1(self, x_tuple):
        x1, x2 = x_tuple
        return (x1)**2 - (x2)**2
    
    def single_saddle2(self, x_tuple):
        x1, x2 = x_tuple
        return x1*x2

    def rosenbrock(self, x_tuple):
        x1, x2 = x_tuple
        return (1 - x1) ** 2 + 100 * (x2 - x1 ** 2) ** 2

    def run_optimizer(self, cost_function, xy_init, optimizer_class, n_iter, **optimizer_kwargs):
        """Run optimization finding the minimum of the Rosenbrock function.
        Parameters
        ----------
        cost_function: function
            rosebrock or other polynomial_cost 
        xy_init : tuple
            Two floats representing the x resp. y coordinates.
        optimizer_class : object
            Optimizer class.
        n_iter : int
            Number of iterations to run the optimization for.
        optimizer_kwargs : dict
            Additional parameters to be passed into the optimizer.
        Returns
        -------
        path : np.ndarray
            2D array of shape `(n_iter + 1, 2)`. Where the rows represent the
            iteration and the columns represent the x resp. y coordinates.
        """
        # print(xy_init)
        xy_t = torch.tensor(xy_init, requires_grad=True)
        optimizer = optimizer_class([xy_t], **optimizer_kwargs)

        path = np.empty((n_iter + 1, 2))
        path[0, :] = xy_init

        for i in tqdm(range(1, n_iter + 1)):
            optimizer.zero_grad()
            loss = cost_function(x_tuple=xy_t) # self.polynomial_cost(x_tuple = xy_t) # rosenbrock(xy_t)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(xy_t, 1.0)
            optimizer.step()

            path[i, :] = xy_t.detach().numpy()

        return path

    def create_animation(self, cost_function, 
                        paths,
                        colors,
                        names,
                        figsize=(12, 12),
                        x_lim=(-2, 2),
                        y_lim=(-1, 3),
                        n_seconds=5):
        """Create an animation.
        Parameters
        ----------
        cost_function: function
            rosebrock or other polynomial_cost
        paths : list
            List of arrays representing the paths (history of x,y coordinates) the
            optimizer went through.
        colors :  list
            List of strings representing colors for each path.
        names : list
            List of strings representing names for each path.
        figsize : tuple
            Size of the figure.
        x_lim, y_lim : tuple
            Range of the x resp. y axis.
        n_seconds : int
            Number of seconds the animation should last.
        Returns
        -------
        anim : FuncAnimation
            Animation of the paths of all the optimizers.
        """
        if not (len(paths) == len(colors) == len(names)):
            raise ValueError

        path_length = max(len(path) for path in paths)

        n_points = 300
        x = np.linspace(*x_lim, n_points)
        y = np.linspace(*y_lim, n_points)
        X, Y = np.meshgrid(x, y)
        # viz = Visualizer()
        Z = cost_function(x_tuple=[X, Y]) # rosenbrock([X, Y])

        minimum = (1.0, 1.0)

        fig, ax = plt.subplots(figsize=figsize)
        ax.contour(X, Y, Z, 90, cmap="jet")

        scatters = [ax.scatter(None,
                            None,
                            label=label,
                            c=c) for c, label in zip(colors, names)]

        ax.legend(prop={"size": 25})
        ax.plot(*minimum, "rD")

        def animate(i):
            for path, scatter in zip(paths, scatters):
                scatter.set_offsets(path[:i, :])

            ax.set_title(str(i))

        ms_per_frame = 1000 * n_seconds / path_length

        anim = FuncAnimation(fig, animate, frames=path_length, interval=ms_per_frame)

        return anim

    def log_result_for_cost(self, cost_function, xy_inits, result_name):
        # xy_init = (.3, .8) # (.1, .1) # (0.0, 0.0) # (.3, .8)
        n_iter = 1500

        optmzrs = [Adam, SGD, Adagrad, RMSprop, ADAM]
        paths_opt = []
        for i, optm in enumerate(optmzrs):
            # print(xy_inits[i])
            x, y = xy_inits[i]
            tuple_xy = (float(x), float(y))
            paths_opt.append(self.run_optimizer(cost_function, tuple_xy, optm, n_iter, lr=1e-3))

        freq = 10

        paths = []
        for path_opt in paths_opt:
            paths.append(path_opt[::freq])

        # paths = [path_adam[::freq], path_sgd[::freq], path_adagrad[::freq], path_rmsprop[::freq], path_ourAdam[::freq]]
        colors = ["green", "blue", "orange", "yellow", 'black']
        names = ["Adam", "SGD", 'Adagrad', 'RMSprop', "Our_ADAM"]

        anim = self.create_animation(cost_function,
                                paths,
                                colors,
                                names,
                                figsize=(12, 7),
                                x_lim=(-4.1, 4.1),  # x_lim=(-.1, 1.1),
                                y_lim=(-4.1, 4.1), # y_lim=(-.1, 1.1),
                                n_seconds=7)

        anim.save("results/result_{}.gif".format(result_name))

    def log_results(self, result_num):
        cost_functions = [self.two_minima_eq, self.two_minima_ueq, self.single_saddle1, self.single_saddle2, self.rosenbrock]
        function_names = ['two_minima_eq', 'two_minima_ueq', 'single_saddle1', 'single_saddle2', 'rosenbrock']
        xy_inits = [[(0.0, 0.0), (0.25, -0.25), (0.5, -0.5), (-0.25, 0.25), (-0.5, 0.5)],
                    [(0.0, 0.0), (0.25, -0.25), (0.5, -0.5), (-0.25, 0.25), (-0.5, 0.5)],
                    [(0.5, -0.5), (0.5, -0.5), (0.5, -0.5), (-0.5, 0.5), (-0.5, 0.5)],
                    [(0.5, -0.5), (0.5, -0.5), (0.5, -0.5), (-0.5, 0.5), (-0.5, 0.5)],
                    [(0.3, 0.8), (0.3, 0.8), (0.3, 0.8), (0.3, 0.8), (0.3, 0.8)]]

        for i, cost_function in enumerate(cost_functions):
            print(xy_inits[i])
            self.log_result_for_cost(cost_function, xy_inits[i], function_names[i]+result_num)


if __name__=='__main__':
    viz = Visualizer()
    viz.log_results(result_num='0')
