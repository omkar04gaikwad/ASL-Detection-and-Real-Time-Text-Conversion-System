{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 43,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/",
     "height": 36
    },
    "id": "pHVJ3h0EoyZE",
    "outputId": "05290559-093b-4623-91ee-9fb92d512b74"
   },
   "outputs": [],
   "source": [
    "import tensorflow as tf\n",
    "from tensorflow.keras.models import Sequential\n",
    "from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D\n",
    "from sklearn.model_selection import train_test_split\n",
    "import numpy as np\n",
    "import pandas as pd"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 44,
   "metadata": {
    "id": "iDC6Ny0wZOWR"
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>alphabets</th>\n",
       "      <th>landmark_0</th>\n",
       "      <th>landmark_1</th>\n",
       "      <th>landmark_2</th>\n",
       "      <th>landmark_3</th>\n",
       "      <th>landmark_4</th>\n",
       "      <th>landmark_5</th>\n",
       "      <th>landmark_6</th>\n",
       "      <th>landmark_7</th>\n",
       "      <th>landmark_8</th>\n",
       "      <th>...</th>\n",
       "      <th>landmark_7990</th>\n",
       "      <th>landmark_7991</th>\n",
       "      <th>landmark_7992</th>\n",
       "      <th>landmark_7993</th>\n",
       "      <th>landmark_7994</th>\n",
       "      <th>landmark_7995</th>\n",
       "      <th>landmark_7996</th>\n",
       "      <th>landmark_7997</th>\n",
       "      <th>landmark_7998</th>\n",
       "      <th>landmark_7999</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>a</td>\n",
       "      <td>[0.4838511645793915, 0.6284142732620239, 0.369...</td>\n",
       "      <td>[0.48559656739234924, 0.6370050311088562, 0.37...</td>\n",
       "      <td>[0.4946433901786804, 0.6508243083953857, 0.376...</td>\n",
       "      <td>[0.48447081446647644, 0.6698311567306519, 0.37...</td>\n",
       "      <td>[0.49145880341529846, 0.6666201949119568, 0.36...</td>\n",
       "      <td>[0.4821345806121826, 0.6677781343460083, 0.358...</td>\n",
       "      <td>[0.48449715971946716, 0.6721535325050354, 0.35...</td>\n",
       "      <td>[0.4841638207435608, 0.6722722053527832, 0.357...</td>\n",
       "      <td>[0.4823983311653137, 0.671943187713623, 0.3552...</td>\n",
       "      <td>...</td>\n",
       "      <td>[0.489443838596344, 0.6730363368988037, 0.3607...</td>\n",
       "      <td>[0.4863154888153076, 0.6711098551750183, 0.361...</td>\n",
       "      <td>[0.4835115075111389, 0.6675708889961243, 0.361...</td>\n",
       "      <td>[0.48151394724845886, 0.667807936668396, 0.359...</td>\n",
       "      <td>[0.486027330160141, 0.6649206280708313, 0.3578...</td>\n",
       "      <td>[0.48650282621383667, 0.6637972593307495, 0.35...</td>\n",
       "      <td>[0.47774815559387207, 0.6650660037994385, 0.35...</td>\n",
       "      <td>[0.47892794013023376, 0.6657238006591797, 0.35...</td>\n",
       "      <td>[0.4793429970741272, 0.6666581630706787, 0.356...</td>\n",
       "      <td>[0.48166096210479736, 0.6673048734664917, 0.35...</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>1 rows × 8001 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "  alphabets                                         landmark_0  \\\n",
       "0         a  [0.4838511645793915, 0.6284142732620239, 0.369...   \n",
       "\n",
       "                                          landmark_1  \\\n",
       "0  [0.48559656739234924, 0.6370050311088562, 0.37...   \n",
       "\n",
       "                                          landmark_2  \\\n",
       "0  [0.4946433901786804, 0.6508243083953857, 0.376...   \n",
       "\n",
       "                                          landmark_3  \\\n",
       "0  [0.48447081446647644, 0.6698311567306519, 0.37...   \n",
       "\n",
       "                                          landmark_4  \\\n",
       "0  [0.49145880341529846, 0.6666201949119568, 0.36...   \n",
       "\n",
       "                                          landmark_5  \\\n",
       "0  [0.4821345806121826, 0.6677781343460083, 0.358...   \n",
       "\n",
       "                                          landmark_6  \\\n",
       "0  [0.48449715971946716, 0.6721535325050354, 0.35...   \n",
       "\n",
       "                                          landmark_7  \\\n",
       "0  [0.4841638207435608, 0.6722722053527832, 0.357...   \n",
       "\n",
       "                                          landmark_8  ...  \\\n",
       "0  [0.4823983311653137, 0.671943187713623, 0.3552...  ...   \n",
       "\n",
       "                                       landmark_7990  \\\n",
       "0  [0.489443838596344, 0.6730363368988037, 0.3607...   \n",
       "\n",
       "                                       landmark_7991  \\\n",
       "0  [0.4863154888153076, 0.6711098551750183, 0.361...   \n",
       "\n",
       "                                       landmark_7992  \\\n",
       "0  [0.4835115075111389, 0.6675708889961243, 0.361...   \n",
       "\n",
       "                                       landmark_7993  \\\n",
       "0  [0.48151394724845886, 0.667807936668396, 0.359...   \n",
       "\n",
       "                                       landmark_7994  \\\n",
       "0  [0.486027330160141, 0.6649206280708313, 0.3578...   \n",
       "\n",
       "                                       landmark_7995  \\\n",
       "0  [0.48650282621383667, 0.6637972593307495, 0.35...   \n",
       "\n",
       "                                       landmark_7996  \\\n",
       "0  [0.47774815559387207, 0.6650660037994385, 0.35...   \n",
       "\n",
       "                                       landmark_7997  \\\n",
       "0  [0.47892794013023376, 0.6657238006591797, 0.35...   \n",
       "\n",
       "                                       landmark_7998  \\\n",
       "0  [0.4793429970741272, 0.6666581630706787, 0.356...   \n",
       "\n",
       "                                       landmark_7999  \n",
       "0  [0.48166096210479736, 0.6673048734664917, 0.35...  \n",
       "\n",
       "[1 rows x 8001 columns]"
      ]
     },
     "execution_count": 44,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "dataset = pd.read_parquet('train_data.parquet')\n",
    "dataset.head(1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 45,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(26, 8000)"
      ]
     },
     "execution_count": 45,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "X = dataset.drop('alphabets', axis=1).values\n",
    "X.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 46,
   "metadata": {},
   "outputs": [],
   "source": [
    "y = pd.get_dummies(dataset['alphabets']).values"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 40,
   "metadata": {},
   "outputs": [],
   "source": [
    "X = X.reshape(int(X.size/len(X)),len(X))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 47,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(26, 8000)"
      ]
     },
     "execution_count": 47,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "X.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 48,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(8, 8000)"
      ]
     },
     "execution_count": 48,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "accelerator": "GPU",
  "colab": {
   "gpuType": "T4",
   "provenance": []
  },
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.13"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
