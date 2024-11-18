#!/bin/bash

# Nombre del trabajo
#SBATCH --job-name=custom-cnn_cs
#SBATCH --output=custom-cnn_cs_%j.log  # Archivo de salida
#SBATCH --error=custom-cnn_cs_%j.err   # Archivo de error
#SBATCH --partition=CPU                  # Partición del cluster
#SBATCH --ntasks=1                       # Número de tareas
#SBATCH --cpus-per-task=4                # Número de CPUs por tarea
#SBATCH --mem=16G                        # Memoria
#SBATCH --time=24:00:00                  # Tiempo máximo de ejecución

# Cargar módulos necesarios
module load anaconda/2021.05
module load cuda/11.3
module load cudnn/8.2.1

# Crear un entorno virtual si no existe
if [ ! -d "$HOME/envs/adam_env" ]; then
    conda create -n adam_env python=3.8 -y
fi

# Activar el entorno virtual
source activate adam_env

# Instalar dependencias si no están instaladas
pip install --no-cache-dir torch torchvision scikit-learn scipy seaborn matplotlib

# Ejecutar el script
python -u GWO_ESQUEMA2.py


# Ejecutar el script y redirigir la salida y el error
#python custom-cnn_cs_binario.py > custom_cnn_cs_%j.log 2> custom_cnn_cs_%j.err