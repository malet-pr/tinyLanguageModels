import matplotlib.pyplot as plt

losses = [] # Populate this in your loop

def save_loss_plot(losses):
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.title("MoE Training Loss")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.savefig("loss_curve.png")
    plt.close()