import os

def export_brevitas_report(report, process_start_time):

    folder_path = "Brevitas Reports"

    if not os.path.exists(folder_path): # Check if the folder exists, if not, create it
        os.makedirs(folder_path)

    report_file_path = os.path.join(folder_path, f"{process_start_time} Brevitas Report.txt")

    with open(report_file_path, "w") as file: # Write the report to a text file
        file.write(report)

    print("Brevitas Report is exported successfully to:", report_file_path)
