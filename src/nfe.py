import io
import json
import streamlit as st
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from google.oauth2 import service_account

FOLDER_ID_NFE = st.secrets["FOLDER_ID_NFE"]

def _load_credentials_json():
    if "GOOGLE_CREDENTIALS" in st.secrets:
        creds_json = st.secrets["GOOGLE_CREDENTIALS"]
        return json.loads(creds_json)
    raise RuntimeError("Credenciais não encontradas em st.secrets.")

@st.cache_resource
def carregar_servico():
    """Carrega o serviço do Google Drive com credenciais seguras."""
    creds_dict = _load_credentials_json()
    creds = service_account.Credentials.from_service_account_info(
        creds_dict,
        scopes=["https://www.googleapis.com/auth/drive"]
    )
    return build("drive", "v3", credentials=creds)

def listar_arquivos_pdfs():
    """Lista os arquivos PDF da pasta do Drive em ordem decrescente."""
    service = carregar_servico()
    results = service.files().list(
        q=f"'{FOLDER_ID_NFE}' in parents and mimeType='application/pdf'",
        orderBy="name desc",
        fields="files(id, name, modifiedTime)"
    ).execute()
    return results.get("files", [])

def baixar_pdf(file_id):
    """Baixa o PDF do Google Drive e retorna como BytesIO."""
    service = carregar_servico()
    request = service.files().get_media(fileId=file_id)
    fh = io.BytesIO()
    downloader = MediaIoBaseDownload(fh, request)

    done = False
    while not done:
        _, done = downloader.next_chunk()

    fh.seek(0)
    return fh