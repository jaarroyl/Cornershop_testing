import chardet
import os.path
import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelEncoder
import seaborn as sb


class color:
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    DARKCYAN = '\033[36m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'


def read_dataset(file):
    
    #Revisar tipo de codificación del archivo csv
    
    with open(file, 'rb') as rawdata:
        result = chardet.detect(rawdata.read(100000))
    #return result

    encoding = result.get('encoding')

    #Obtener el tipo de extension del archivo
    
    extension = os.path.splitext(file)[1]
    
    #Lectura del archivo de acuerdo a la extension
    if extension == '.csv':
         
        print('[INFO]... Leyendo csv con separador ";" 💾') 
        dataset = pd.read_csv(file, delimiter=";", decimal= '.', thousands= ',', encoding= encoding, low_memory= False)
        if dataset.columns.value_counts().values[0] <= 1: #Se puede mejorar utilizando libreria que lea el tipo de separador que trae el csv
            print('[INFO]... Leyendo csv con otro separador 💾')
            dataset = pd.read_csv(file, delimiter=",",  decimal= '.', thousands= ',', encoding= encoding, low_memory= False)

    else:

        print('[INFO]... Leyendo excel 💾')
        dataset = pd.read_excel(file,  decimal= '.', thousands= ',')
    
    return dataset


def estadisticos(data_dict):

    for i in data_dict.keys():
        
        print(color.BOLD + color.RED  + '\n[INFO]... Analizando base {}: 🔍 \n '.format(i) + color.END + color.END)
        
        print('[INFO]... Columnas del dataset: 👾\n')
        print(data_dict[i].columns.tolist()) 

        print('\n[INFO]... Cantidad de datos del dataset: 📉') 
        print('\nNumero de columnas: ', data_dict[i].shape[1])
        print('\nNumero de filas: ', data_dict[i].shape[0])
        
        print('\n[INFO]... Cantidad de datos nulos por columna: 🚫\n')
        print(data_dict[i].isnull().sum())

        print('\n[INFO]... Tipos de variables: 🪴\n')
        print(data_dict[i].dtypes)
        
        print('\n[INFO]... Estadisticos basicos: 🎯 \n ') 
        print(data_dict[i].describe())


#Funcion para graficar
def plot_hist_frec(df, cols):
        
    plt.figure(figsize=(16,8));
    rows = df.shape[1] / (cols - 1)
    
    for index, (colname, serie) in enumerate(df.iteritems()):
        plt.subplot(rows, cols, index + 1)
        if pd.api.types.is_float_dtype(serie) is True:
            sb.distplot(serie, color='blue')
            plt.axvline(np.mean(serie), color='tomato')
        elif pd.api.types.is_integer_dtype(serie) is True:
            sb.countplot(serie, color='blue')
        plt.title(colname, fontsize=16)
        plt.xlabel('');
        plt.ylabel('');
        plt.tight_layout();



#Generar grafico para revisar outliers
def plot_boxplot(dataset, num_cols, column_list):


    print('[INFO] Revisando outliers en campos tipo numericos... 🔍\n')
    x = dataset[column_list]
    
    plt.figure(figsize=(25,10));
    rows = x.shape[1] / (num_cols - 1)
    
    for index, (colname, serie) in enumerate(x.iteritems()):
        plt.subplot(rows, num_cols, index + 1)

        if pd.api.types.is_float_dtype(serie) is True:
            sb.boxplot(serie, color='blue')
            plt.axvline(np.mean(serie), color='tomato')

        elif pd.api.types.is_integer_dtype(serie) is True:
            sb.boxplot(serie, color='blue')
        plt.title(colname, fontsize=20)
        plt.xlabel('');
        plt.ylabel('');
        plt.tight_layout();


#Funcion para correlacion
def correlation(dataset):
    
    sb.set(style="white")
    print('[INFO] Generando grafico de correlacion... 🚀\n')
    corr = dataset.loc[:,:].corr()
    
    # Generate a mask for the upper triangle
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(11, 9))

    # Generate a custom diverging colormap
    cmap = sb.diverging_palette(600, 10, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    sb.heatmap(corr, mask=mask,cmap=cmap, vmax=.3, center=0,
        square=True, linewidths=.5, cbar_kws={"shrink": .5})


#Frecuencia de valores unicos
def unique_values(dataset, threshold, columns_list):

    for x in columns_list:
        if len(dataset[x].unique()) > threshold:
            print(color.BOLD + '\n[INFO] Frecuencia por dato unico para campo {} 🚀...'.format(x) + color.END)
            print('\nCantidad de valores unicos por categoría: {}'.format(len(dataset[x].unique())))
            print('\nMostrando el top {} categorias según frecuencia '.format(threshold))
            data = {
                'Cantidad': dataset[x].groupby(dataset[x]).count().sort_values(ascending=False),
                'Frecuencia': (dataset[x].value_counts(normalize=True).sort_values(ascending=False)*100)
                 }
            y = pd.DataFrame(data) #Crear dataframe con dos columnas, cantidad y frecuencia para mostrar en conjunto
            y = y.sort_values('Cantidad',ascending=False)
            y['Frecuencia Acumulada'] = y['Frecuencia'].cumsum() #Agregar frecuencia acumulada
            print(y.head(threshold))
        else:
            print(color.BOLD + '\n[INFO]\nFrecuencia por dato unico para campo {} 🚀...'.format(x) + color.END)
            data = {
                'Cantidad': dataset[x].groupby(dataset[x]).count().sort_values(ascending=False),
                'Frecuencia': (dataset[x].value_counts(normalize=True).sort_values(ascending=False)*100)
                }
            y = pd.DataFrame(data )#Crear dataframe con dos columnas, cantidad y frecuencia para mostrar en conjunto
            y = y.sort_values('Cantidad',ascending=False)
            y['Frecuencia Acumulada'] = y['Frecuencia'].cumsum() #Agregar frecuencia acumulada
            print(y)


#Funcion para encoding de variables
def cat_encoding(dataset):
    
    print('\n[INFO] Transformando variables... 🚀')
    df_2 = dataset.copy()
    list = df_2.select_dtypes(include=['object']).columns.to_list()
    for i in range(len(list)):
        df_2[str(list[i]) + '_cat'] = LabelEncoder().fit_transform(df_2[list[i]]).astype(object)
    
    print('[INFO] Seleccionando variables... ✅\n')
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64', 'object']
    dataset_out = df_2.select_dtypes(include=numerics).copy()
    print('[INFO] Dataset final: ... 🛫\n')
    return dataset_out
