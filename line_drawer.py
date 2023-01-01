import sys
import matplotlib.pyplot as plt


class LineDrawer:
  def __init__(self, fig, ax, on=True, func=None) -> None:
    self.ax = ax
    self.fig = fig
    self.xs = []
    self.ys = []
    self.line = None
    if on:
      self.cid = fig.canvas.mpl_connect('button_press_event', self)
    else:
      self.cid = None
    self.on = on
    if func:
      self.func = func
    else:
      self.func = lambda x: x

  def __call__(self, event):
    xlim = self.ax.get_xlim()
    ylim = self.ax.get_ylim()
    if len(self.xs) == 2:
      self.xs = []
      self.ys = []
      self.ax.cla()
      self.ax.set_xlim(xlim)
      self.ax.set_ylim(ylim)
    self.xs.append(event.xdata)
    self.ys.append(event.ydata)
    self.line = self.ax.plot(self.xs, self.ys)
    self.func(self)
    self.fig.canvas.draw()

  def __setattr__(self, __name: str, __value) -> None:
    self.__dict__[__name] = __value
    if __name == 'on':
      if self.on:
        self.cid = self.fig.canvas.mpl_connect('button_press_event', self)
      else:
        if self.cid:
          self.fig.canvas.mpl_disconnect(self.cid)
          self.cid = None


class LineDrawerX(LineDrawer):
  def __init__(self, fig, ax, on=True, func=None) -> None:
    super().__init__(fig, ax, on, func)

  def __call__(self, event):
    xlim = self.ax.get_xlim()
    ylim = self.ax.get_ylim()
    self.ax.cla()
    self.ax.set_xlim(xlim)
    self.ax.set_ylim(ylim)
    self.ys = []
    self.ys.append(event.ydata)
    self.ys.append(event.ydata)
    xlim = list(xlim)
    self.line = self.ax.plot(xlim, self.ys)
    self.func(self)
    self.fig.canvas.draw()

  def __setattr__(self, __name: str, __value) -> None:
    return super().__setattr__(__name, __value)


class LineDrawerY(LineDrawer):
  def __init__(self, fig, ax, on=True, func=None) -> None:
    super().__init__(fig, ax, on, func)

  def __call__(self, event):
    xlim = self.ax.get_xlim()
    ylim = self.ax.get_ylim()
    self.ax.cla()
    self.ax.set_xlim(xlim)
    self.ax.set_ylim(ylim)
    self.xs = []
    self.xs.append(event.xdata)
    self.xs.append(event.xdata)
    ylim = list(ylim)
    self.line = self.ax.plot(self.xs, ylim)
    self.func(self)
    self.fig.canvas.draw()

  def __setattr__(self, __name: str, __value) -> None:
    return super().__setattr__(__name, __value)


if __name__ == '__main__':
  fig, ax = plt.subplots(1, 2)
  for a in ax:
    a.set_xlim((0., 4.))
    a.set_ylim((0., 4.))
  line_drawer = LineDrawer(fig, ax[0])
  line_drawer_x = LineDrawerX(fig, ax[0], on=False)
  line_drawer_y = LineDrawerY(fig, ax[0], on=False)
  def on_key_press(event):
    sys.stdout.flush()
    if event.key == 'n':
      line_drawer_x.on = False
      line_drawer_y.on = False
      line_drawer.on = True
    elif event.key == 'x':
      line_drawer_x.on = True
      line_drawer_y.on = False
      line_drawer.on = False
    elif event.key == 'y':
      line_drawer_x.on = False
      line_drawer_y.on = True
      line_drawer.on = False
    elif event.key == 'q':
      plt.close()
  fig.canvas.mpl_connect('key_press_event', on_key_press)

  plt.show()
