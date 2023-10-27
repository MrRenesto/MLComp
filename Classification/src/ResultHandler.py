import pandas as pd
from datetime import datetime
import subprocess


def upload_result(validation_data, y_pred_val, message):

    user_input = input("Do you want to build Model and Submit Results? Y/N: ")

    if user_input == "Y" or user_input == "y":
        # 4. Create a DataFrame with the predictions and the 'Id' column
        predictions_df = pd.DataFrame({'Id': validation_data['Id'], 'Predicted_Label': y_pred_val})

        # 5. Save the predictions to a CSV file
        current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_name = f'predictions_validation_{current_datetime}.csv'

        # Save the DataFrame to the CSV file
        predictions_df.to_csv("Predictions/" + file_name, index=False)

        message = message.replace("\n", " ")
        command = "kaggle competitions submit -c seminar-isg-ml-competition-ws23-classification -m \"" + message + "\" -f Predictions/" + file_name
        try:
            # Run the command and capture its output
            result = subprocess.check_output(command, shell=True, universal_newlines=True)

            # Print the output
            print(result)
        except subprocess.CalledProcessError as e:
            # If the command returns a non-zero exit status, an exception is raised
            print("Command failed with exit code", e.returncode)
        #

    else:
        print("Did not Submit anything")

