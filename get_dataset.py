import pandas as pd
import os
import requests

def download_file(url, save_path):
    for i in range(5):
        try:
            # Send the GET request to fetch the file
            response = requests.get(url)

            # Check if the request was successful (status code 200)
            if response.status_code == 200:
                with open(save_path, 'wb') as file:
                    file.write(response.content)
                print(f"File downloaded successfully and saved to: {save_path}")
                break

            else:
                print(f"Failed to download the file {save_path}. Status code: {response.status_code}")
        
        except Exception as e:
            print(f"Failed as an error ocurred {save_path}: {e}")

if __name__ == '__main__':
    save_path = "dataset"  # change if needed
    df = pd.read_csv('labels.csv') # change if needed


    # IMPORTANT: change the authtoken from the website 
    auth_token = ''

    for index, row in df.iterrows():
        patient_number = row['No.']
        
        for image_number in range(1, 14):  # Images 1 to 13
            image_column = f'Image {image_number}'
            patient_image_url = row[image_column]
            
            if patient_image_url != '-':  # Only process URLs that are not '-'
                folder_path = os.path.join(save_path, str(patient_number))
                if not os.path.exists(folder_path):
                    os.makedirs(folder_path)

                uid = patient_image_url.split('/')[-1]

                final_url = f'https://site-9-production.arogyam-midas.iisc.ac.in/fhir/DocumentReference/{uid}/$binary-access-read?path=DocumentReference.content.attachment&&authToken={auth_token}'
                final_path = os.path.join(folder_path, f'image_{image_number}.jpeg')
                download_file(final_url, final_path)

    # print("Image download process completed.")
