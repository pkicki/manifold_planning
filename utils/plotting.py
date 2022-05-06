import matplotlib.pyplot as plt


def plot_qs(idx, q, q_dot, q_ddot, dt):
    for i in range(6):
        plt.plot(dt[0], q[0, :, i], label=f"q_{i}")
    plt.legend()
    plt.savefig(f"q_{idx:05d}.png")
    plt.clf()
    for i in range(6):
        plt.plot(dt[0], q_dot[0, :, i], label=f"q_dot_{i}")
    plt.legend()
    plt.savefig(f"q_dot_{idx:05d}.png")
    plt.clf()
    for i in range(6):
        plt.plot(dt[0], q_ddot[0, :, i], label=f"q_ddot_{i}")
    plt.legend()
    plt.savefig(f"q_ddot_{idx:05d}.png")
    plt.clf()

