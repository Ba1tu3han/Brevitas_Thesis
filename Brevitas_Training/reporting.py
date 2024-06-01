import os
import matplotlib.pyplot as plt

def export_brevitas_report(report, process_start_time):

    folder_path = "Brevitas Reports"

    if not os.path.exists(folder_path): # Check if the folder exists, if not, create it
        os.makedirs(folder_path)

    report_file_path = os.path.join(folder_path, f"{process_start_time} Brevitas Report.txt")

    with open(report_file_path, "w") as file: # Write the report to a text file
        file.write(report)

    print("Brevitas Report is exported successfully to:", report_file_path)


def export_accuracy_graph(train_losses, test_losses, train_accuracies, test_accuracies, process_start_time, t):

    accuracy_graph_folder_path = "Accuracy Graphs"

    if not os.path.exists(accuracy_graph_folder_path): # Check if the folder exists, if not, create it
        os.makedirs(accuracy_graph_folder_path)

    epoch_list = list(range(t + 1))  # to list epochs to the end
    fig, ax = plt.subplots(2, 1)

    ax[0].plot(epoch_list, train_losses, label="Train Loss")
    ax[0].plot(epoch_list, test_losses, label="Validation Loss")
    ax[0].set_ylabel('Loss')
    ax[0].legend()

    ax[1].plot(epoch_list, train_accuracies, label="Train Accuracy")
    ax[1].plot(epoch_list, test_accuracies, label="Validation Accuracy")
    ax[1].set_xlabel('Number of Epoch')
    ax[1].set_ylabel('Accuracy (Top1)')
    ax[1].legend()

    figure_name = "Accuracy_Loss_Plot " + str(process_start_time) + ".png"
    plt.savefig(os.path.join(accuracy_graph_folder_path, figure_name))

    print("Accuracy Graph is exported successfully to:", accuracy_graph_folder_path)
