from sentence_transformers import SentenceTransformer, util
import numpy as np
#import matplotlib.pyplot as plt
#from sklearn.metrics.pairwise import cosine_similarity
import os
import pandas as pd

# List of models optimized for semantic textual similarity can be found at:
# https://docs.google.com/spreadsheets/d/14QplCdTCDwEmTqrn1LH4yrbKvdogK4oQvYO1K1aPR5M/edit#gid=0
#distiluse-base-multilingual-cased-v1
#model = SentenceTransformer('stsb-roberta-large')
#https://www.sbert.net/docs/pretrained_models.html
model = SentenceTransformer('distiluse-base-multilingual-cased-v1')



def get_filenames():

	#get the directory where patients folders are stored
	image_directory = 'data/'
	#get patients folder names 
	patient_names = os.listdir(image_directory)

	#set empty arrays to fill with filenames and files content
	descriptions = []
	photo_names = []
	photo_titles = []


	#explore patient folders and fill arrays according to value (image, title or text)
	for patient_name in patient_names:
	    dir_files = os.listdir(image_directory+patient_name)
	    temp_list = []
	    temp_list_photos = []
	    temp_list_titles = []
	    for dir_file in dir_files:
	        if (dir_file.split('_')[1]=="photo"):
	            #print(dir_file)
	            if(dir_file.split('_')[3]=="text.txt"):
	                #print(dir_file)
	                desc = open(image_directory+patient_name+"/"+dir_file, encoding='utf-8').read().rstrip('\n')
	                temp_list.append(desc)
	            elif(dir_file.split('_')[3]=="image.png"):
	                temp_list_photos.append(dir_file)
	            elif(dir_file.split('_')[3]=="title.txt"):
	                title = open(image_directory+patient_name+"/"+dir_file, encoding='utf-8').read().rstrip('\n')
	                temp_list_titles.append(title)                             
	                
	    descriptions.append(temp_list)
	    photo_names.append(temp_list_photos)
	    photo_titles.append(temp_list_titles)

	    return(patient_names, descriptions, photo_names, photo_titles)



def create_dataframe(patient_names, array_to_df):
	df = pd.DataFrame(array_to_df).transpose()
	df.columns = patient_names
	return df


def get_similarity(patient_1, patient_2, df_texts, df_photos, df_titles):
    embeddings_patient_1 = np.empty([df_texts[patient_1].dropna().size,512])
    embeddings_patient_2 = np.empty([df_texts[patient_2].dropna().size,512])
    sim_matrix = np.empty([10,10])
    for i in range(df[patient_1].dropna().size):
        sentence_embedding = model.encode(df_texts[patient_1].dropna().iloc[i], convert_to_tensor=True)
        embeddings_patient_1[i] = sentence_embedding

    for i in range(df[patient_2].dropna().size):
        sentence_embedding = model.encode(df_texts[patient_2].dropna().iloc[i], convert_to_tensor=True)
        embeddings_patient_2[i] = sentence_embedding

    # compute similarity scores of two embeddings
	cosine_scores = util.pytorch_cos_sim(embeddings_patient_1, embeddings_patient_2)

	for i in range(len(embeddings_patient_1)):
	    for j in range(len(embeddings_patient_2)):
	        if(cosine_scores[i][j].item() == cosine_scores.max().item()):
	            #print("Description", patient_1, ":", df_texts['patient_1'].dropna().iloc[i])
	            #print("Description", patient_2, ":", df_texts['patient_2'].dropna().iloc[j])
	            #print("Similarity Score:", cosine_scores[i][j].item())
	            index_1 = i
	            index_2 = j
	            
	#print(index_1)
	#print(index_2)
	photo_file1 = df_photos[patient_1].dropna().iloc[index_1]
	photo_title1 = df_titles[patient_1].dropna().iloc[index_1]

	photo_file2 = df_photos[patient_2].dropna().iloc[index_2]
	photo_title2 = df_titles[patient_2].dropna().iloc[index_2]

	#print(photo_file1)
	#print(photo_title1)
	#print(photo_file2)
	#print(photo_title2)

	return (photo_file1, photo_title1, photo_file2, photo_title2)



def get_similar_files(patient_1, patient_2):

	patient_names, descriptions, photo_names, photo_titles = get_filenames()
	df_texts = create_dataframe(patient_names, descriptions)
	df_photos = create_dataframe(patient_names, photo_names)
	df_titles = create_dataframe(patient_names, photo_titles)
	photo_file1, photo_title1, photo_file2, photo_title2 = get_similarity(patient_1, patient_2, df_texts, df_photos, df_titles)
	return (photo_file1, photo_title1, photo_file2, photo_title2)


#patient_1 = "VINCENT"
#patient_2 = "PETER"

#photo_file1, photo_title1, photo_file2, photo_title2 = get_similar_files(patient_1, patient_2)
