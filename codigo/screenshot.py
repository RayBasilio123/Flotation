import pyautogui
import time
import os
import msvcrt
from datetime import datetime
import argparse

# Configuração do parser de argumentos para receber o diretório de saída
parser = argparse.ArgumentParser(description='Captura de screenshots a cada 20s.')
parser.add_argument('--output_dir', type=str, default=r"c:\Users\rayba\Music\mestrado_dados\vale-nova\imagens",
                    help='Caminho para salvar as screenshots')
args = parser.parse_args()

output_dir = args.output_dir

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

try:
    while True:
        # Verifica se alguma tecla foi pressionada
        if msvcrt.kbhit():
            key = msvcrt.getch()
            # Se a tecla for 'p' (minúsculo), interrompe o loop
            if key == b'p':
                print("Encerrado pelo usuário.")
                break

        # Gera um nome baseado na data e hora atual
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        file_name = f"screenshot_{timestamp}.png"
        full_path = os.path.join(output_dir, file_name)
        
        # Tira a screenshot e salva
        screenshot = pyautogui.screenshot()
        screenshot.save(full_path)
        print(f"Screenshot salva: {full_path}")
        
        # Aguarda 20 segundos
        time.sleep(20)
except Exception as e:
    print("Erro:", e)