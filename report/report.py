import multiprocessing

from matplotlib import pyplot as plt


class OpenPlot:
    @staticmethod
    def open_plot(fig):
        plt.close(fig)
        OpenPlot.show_figure(fig)
        plt.show()

    @staticmethod
    def show_figure(fig):
        # create a dummy figure and use its
        # manager to display "fig"

        dummy = plt.figure()
        new_manager = dummy.canvas.manager
        new_manager.canvas.figure = fig
        fig.set_canvas(new_manager.canvas)

    @staticmethod
    def run_proc(fig):
        multiprocessing.Process(target=OpenPlot.open_plot, args=(fig,)).start()