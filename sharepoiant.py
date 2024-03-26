import requests

# Öncelikle erişim token'ınızı alın
token = "ERIŞIM_TOKENI"

# Dosyanın indirileceği SharePoint dosya URL'i
file_url = "https://divan-my.sharepoint.com/:x:/g/personal/tolga_turan_divan_com_tr/ERpbGiXr1rROoZFFsB8fbiMBTUoZh0Zq9c4wrCdyTcf5RA?e=MLaynf"

# Dosyanın kaydedileceği yerel yol
target_path = "kitap2.xlsx"

headers = {
    'Authorization': f'Bearer {token}'
}

response = requests.get(file_url, headers=headers, stream=True)

if response.status_code == 200:
    with open(target_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=1024):
            f.write(chunk)
    print("Dosya başarıyla indirildi:", target_path)
else:
    print("Dosya indirme başarısız. Hata kodu:", response.status_code)
