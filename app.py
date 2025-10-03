import streamlit as st 
import pandas as pd 
import plotly.express as px 
import plotly.graph_objects as go 
import numpy as np
from src.nfe import listar_arquivos_pdfs, baixar_pdf
import src.utils as util

st.set_page_config(
    layout="wide",
    page_title="Gestão e Análise | Rafael Trindade", 
    initial_sidebar_state="expanded", 
    page_icon="📊")

sidebar_logo = "https://i.postimg.cc/yxNK4Cxs/logo-rafael.png"
main_body_logo = "https://i.postimg.cc/3xkGPmC6/streamlit02.png"
st.logo(sidebar_logo, icon_image=main_body_logo)

util.aplicar_estilo()

CODE = st.secrets["CODE"]

EMPRESA_FILES = st.secrets["EMPRESA_FILES"]
nomes_empresas = [info["nome"] for info in EMPRESA_FILES.values()]
empresa_nome = st.sidebar.selectbox("Selecione a Empresa:", nomes_empresas)
empresa_chave = next(k for k, v in EMPRESA_FILES.items() if v["nome"] == empresa_nome)

def gdrive_csv(file_id: str) -> str:
    return f"https://docs.google.com/spreadsheets/d/{file_id}/export?format=csv"

def carregar_csv(tipo: str) -> pd.DataFrame:
    arquivos = EMPRESA_FILES[empresa_chave]
    nome_empresa = arquivos["nome"]
    try:
        df = pd.read_csv(
            gdrive_csv(arquivos[tipo]),
            sep=",", decimal=",", thousands=".",
            index_col=None
        )
        if 'data' in df.columns:
            df['data'] = pd.to_datetime(df['data'], errors='coerce', dayfirst=True)
        df['empresa'] = nome_empresa
        return df
    except Exception as e:
        st.warning(f"Erro ao carregar {tipo} de {nome_empresa}: {e}")
        return pd.DataFrame()

df_faturamento = carregar_csv("faturamento")
df_despesa = carregar_csv("despesas")
df_capital = carregar_csv("capital")

col1_side, col2_side = st.sidebar.columns([2,1])
col1_side.markdown('<h5 style="margin-bottom: -25px;">Início Apurado:', unsafe_allow_html=True)
col2_side.markdown('<h5 style="text-align: end; margin-bottom: -25px;">' + str(df_faturamento['data'].min().strftime('%d/%m/%Y'))+ '</h5>', unsafe_allow_html=True)
col1_side.markdown('<h5 style="margin-bottom: 0px; color: #053061;">Última Atualização:</h5>', unsafe_allow_html=True)
col2_side.markdown('<h5 style="margin-bottom: 0px; text-align: end; color: #053061;">' + str(df_faturamento['data'].max().strftime('%d/%m/%Y'))+ '</h5>', unsafe_allow_html=True)


if "autenticado_tab4" not in st.session_state:
    st.session_state["autenticado_tab4"] = False
if "aba_ativa" not in st.session_state:
    st.session_state["aba_ativa"] = "faturamento_diário"

abas = [
    "📅 Faturamento Diário",
    "📊 Faturamento Mensal",
    "📊 Despesa Mensal",
    "⚖️ Resultados | Indicadores"
]

if empresa_nome == "MH Refeições":
    abas.append("📄 NFE's")

tab_objects = st.tabs(abas)

########################################################################################
####### ABA FATURAMENTO DIÁRIO #########################################################
########################################################################################
with tab_objects[0]:
    with st.container(border=True):
        col1, col2,col5 = st.columns([1, 1, 2.02])
        col3, col4 = st.columns([1,1])
    with st.container(border=True):
        col8, col9, c200, c2000 = st.columns([2.5,2,3.5,1])

empresas = sorted(df_faturamento['empresa'].unique())
empresa_nome = col5.selectbox('Selecione a Empresa:', empresas, disabled=True)

########################################################################################
####### TABELA FATURAMENTO DIÁRIO ######################################################
########################################################################################

if empresa_nome == "Pousada da Ponte":

    try:
        anos_disponiveis = util.anos_disponiveis(df_faturamento)

        ano_selecionado = col1.selectbox(
            'Selecione o Ano:',
            anos_disponiveis,
            index=len(anos_disponiveis)-1,
            key="tabela_ano"
        )

        meses_nomes_disponiveis = util.atualiza_meses_disponiveis(ano_selecionado, df_faturamento)

        mes_nome_selecionado = col2.selectbox(
            'Selecione um Mês:',
            meses_nomes_disponiveis,
            index=len(meses_nomes_disponiveis)-1,
            key="tabela_mes"
        )
        mes_selecionado = [
            num for num, nome in util.mapa_meses.items() 
            if nome == mes_nome_selecionado
        ][0]

        df_filtrado = df_faturamento[
            (df_faturamento['data'].dt.year == ano_selecionado) &
            (df_faturamento['data'].dt.month == mes_selecionado)
        ]

        if df_filtrado.empty:
            col3.warning(f"Não há dados disponíveis para {mes_nome_selecionado}/{ano_selecionado}.")
        else:

            grouped = df_filtrado.groupby(['data', 'servico'], as_index=False).sum(numeric_only=True)
            pivot_df = (
                grouped
                .pivot(index='data', columns='servico', values='valor')
                .reset_index()
                .fillna(0)
            )

            for col_faturamento in ['Frigobar', 'Hospedagem']:
                if col_faturamento not in pivot_df.columns:
                    pivot_df[col_faturamento] = 0

            numeric_cols = pivot_df.select_dtypes(include='number').columns.tolist()
            if 'data' in numeric_cols:
                numeric_cols.remove('data')
            pivot_df['TOTAL'] = pivot_df[numeric_cols].sum(axis=1)

            df_hospedes_obs = (
                df_filtrado
                .groupby('data', as_index=False)
                .agg({
                    'qtd_hospedes': 'first',  
                    'obs': 'first'            
                })
            )

            pivot_df = pivot_df.merge(df_hospedes_obs, on='data', how='left')

            pivot_df['data'] = pivot_df['data'].dt.strftime('%d/%m/%Y')
            pivot_df.rename(columns={'data': 'Data'}, inplace=True)

            pivot_df[['Frigobar', 'Hospedagem', 'TOTAL', 'qtd_hospedes']] = pivot_df[[
                'Frigobar', 'Hospedagem', 'TOTAL', 'qtd_hospedes'
            ]].astype(float)

            totals = {
                'Data': 'TOTAL',
                'Frigobar': pivot_df['Frigobar'].sum(),
                'Hospedagem': pivot_df['Hospedagem'].sum(),
                'TOTAL': pivot_df['TOTAL'].sum(),
                'qtd_hospedes': pivot_df['qtd_hospedes'].sum(),
                'obs': '-'
            }

            df_totals = pd.DataFrame([totals])
            pivot_df = pd.concat([df_totals, pivot_df], ignore_index=True)

            def calc_apt(row):
                if row['qtd_hospedes'] == 0:
                    return 0.0
                return row['Hospedagem'] / row['qtd_hospedes']

            pivot_df['apt'] = pivot_df.apply(calc_apt, axis=1)


            pivot_df.rename(columns={
                'qtd_hospedes': 'Qtd. Hóspedes',
                'obs': 'Observação',
                'apt': 'APT'
            }, inplace=True)


            col_order = [
                'Data', 
                'Hospedagem',
                'Frigobar',  
                'Observação', 
                'Qtd. Hóspedes', 
                'TOTAL',
                'APT'
            ]
            for col in col_order:
                if col not in pivot_df.columns:
                    if col == 'Observação': 
                        pivot_df[col] = ''
                    else:
                        pivot_df[col] = 0
            pivot_df = pivot_df[col_order]


            for col_moeda in ['Frigobar', 'Hospedagem', 'TOTAL', 'APT']:
                pivot_df[col_moeda] = pivot_df[col_moeda].apply(util.formata_para_brl)

            pivot_df['Qtd. Hóspedes'] = pivot_df['Qtd. Hóspedes'].apply(
                lambda x: str(int(float(x))) if pd.notnull(x) and str(x).replace('.', '').isdigit() else x
            )

            styled_pivot_df = (
                pivot_df.style
                .apply(util.table_highlight_rows, axis=None)
            )

            col3.dataframe(
                styled_pivot_df,
                use_container_width=True,
                height=352,
                hide_index=True
            )

    except pd.errors.ParserError as e:
        col3.error(f"Erro ao analisar o arquivo CSV: {e}")
    except Exception as e:
        col3.error(f"Ocorreu um erro inesperado: {e}")


elif empresa_nome == "MH Refeições":

    try:
        anos_disponiveis = util.anos_disponiveis(df_faturamento)

        ano_selecionado = col1.selectbox('Selecione o Ano:', anos_disponiveis, index=len(anos_disponiveis)-1, key="tabela_ano2")
        meses_nomes_disponiveis = util.atualiza_meses_disponiveis(ano_selecionado, df_faturamento)
        mes_nome_selecionado = col2.selectbox('Selecione um Mês:', meses_nomes_disponiveis, index=len(meses_nomes_disponiveis)-1, key="tabela_mes2")
        mes_selecionado = [num for num, nome in util.mapa_meses.items() if nome == mes_nome_selecionado][0]

        empresas = sorted(df_faturamento['empresa'].unique())

        df_filtrado = df_faturamento[(df_faturamento['data'].dt.year == ano_selecionado) & 
                                    (df_faturamento['data'].dt.month == mes_selecionado) & 
                                    (df_faturamento['empresa'] == empresa_nome)]
        
        if df_filtrado.empty:
            col3.warning(f"Não há dados disponíveis para **{empresa_nome}** em **{mes_nome_selecionado}/{ano_selecionado}**.")
        else:
            grouped = df_filtrado.groupby(['data', 'servico']).sum(numeric_only=True).reset_index()
            pivot_df = grouped.pivot(index='data', columns='servico', values='valor').reset_index()
            pivot_df = pivot_df.fillna(0)

            service_columns = [col for col in pivot_df.columns if col != 'data']

            pivot_df['TOTAL'] = pivot_df[service_columns].sum(axis=1)

            pivot_df['data'] = pivot_df['data'].dt.strftime('%d/%m/%Y')

            rename_dict = {
                'data':'Data',
            }
            pivot_df = pivot_df.rename(columns=rename_dict)

            col_order = ['Data'] + service_columns + ['TOTAL']
            pivot_df = pivot_df[col_order]

            cols_except_data = [col for col in pivot_df.columns if col != 'Data']
            pivot_df[cols_except_data] = pivot_df[cols_except_data].map(util.formata_para_brl)

            pivot_df_float = pivot_df.copy()
            pivot_df_float[cols_except_data] = pivot_df_float[cols_except_data].map(util.brl_para_float)

            totals = pivot_df_float[cols_except_data].sum(numeric_only=True)
            totals['Data'] = 'TOTAL'

            pivot_df = pd.concat([pivot_df, pd.DataFrame([totals])], ignore_index=True)

            pivot_df = pivot_df.reindex([len(pivot_df)-1] + list(range(len(pivot_df)-1)))

            pivot_df[cols_except_data] = pivot_df[cols_except_data].map(util.formata_para_brl)

            pivot_df = pivot_df.reset_index(drop=True)

            styled_pivot_df = pivot_df.style.apply(util.table_highlight_rows, axis=None)

            col3.dataframe(styled_pivot_df, use_container_width=True, height=352, hide_index=True)

    except pd.errors.ParserError as e:
        col3.error(f"Erro ao analisar o arquivo CSV: {e}")
    except Exception as e:
        col3.error(f"Ocorreu um erro inesperado: {e}")

########################################################################################
####### GRÁFICO BARRAS FATURAMENTO DIÁRIO ##############################################
########################################################################################

with col4:
    col6 = st.container()
    c1 = st.container()

    try:
        if empresa_nome == 'Todos':
            servicos = ['Todos'] + sorted(df_faturamento['servico'].unique())
        else:
            servicos = ['Todos'] + sorted(df_faturamento[df_faturamento['empresa'] == empresa_nome]['servico'].unique())
        selected_servico = col6.selectbox('Selecione o Serviço:', servicos)

        df_filtrado_empresas = df_faturamento.copy()
        if empresa_nome != 'Todas':
            df_filtrado_empresas = df_filtrado_empresas[df_filtrado_empresas['empresa'] == empresa_nome]
        if selected_servico != 'Todos':
            df_filtrado_empresas = df_filtrado_empresas[df_filtrado_empresas['servico'] == selected_servico]

        grouped_diario = df_filtrado_empresas.groupby(df_filtrado['data']).sum(numeric_only=True).reset_index()
        grouped_diario['data_str'] = grouped_diario['data'].dt.strftime('%d/%m')

        fig_faturamento_diario = px.bar(
            grouped_diario, 
            x='data_str', 
            y='valor', 
            orientation='v',
            text=grouped_diario['valor'].apply(util.formata_para_brl),
            color_discrete_sequence=[util.barra_azul],
            labels={'data_str': 'Data', 'valor': 'Valor Total'},

        )

        fig_faturamento_diario.update_yaxes(title_text="", showline=True, linecolor="Grey",linewidth=0.1, gridcolor='lightgrey', dtick=200)
        fig_faturamento_diario.update_xaxes(title_text="", showline=True, linecolor="Grey", linewidth=0.1, gridcolor='lightgrey', title="Período")
        fig_faturamento_diario.update_traces(textposition='inside')
        fig_faturamento_diario.update_layout(margin=dict(t=0, b=0, l=0, r=0),height=276,yaxis_title=f"Faturado ({mes_nome_selecionado}/{ano_selecionado})",yaxis=dict(showticklabels=False))

        c1.plotly_chart(fig_faturamento_diario, use_container_width=True, automargin=True)

    except pd.errors.ParserError as e:
        print(f"Erro ao analisar o arquivo CSV: {e}")
    except Exception as e:
        c1.warning(f"Ocorreu um erro: {e}")


########################################################################################
####### GRÁFICO BOX FATURAMENTO DIÁRIO (GERAL) #########################################
########################################################################################    
    try:
        df_filtrado_empresas_graph = df_filtrado.copy()
        if empresa_nome != 'Todas':
            df_filtrado_empresas_graph = df_filtrado[df_filtrado['empresa'] == empresa_nome]
            x_title = f'{empresa_nome}'
        else:
            df_filtrado_empresas_graph = df_filtrado_empresas_graph
            x_title = "Todas as Empresas"

        specific_color = px.colors.sequential.RdBu_r[1:2] + px.colors.sequential.Bluyl_r[0:]

        main_title = f"-BOX PLOT FATURAMENTO ({mes_nome_selecionado.upper()} DE {ano_selecionado})"
        empresa_part = f"Empresa: {empresa_nome}" if empresa_nome != "Todas" else "Todas as Empresas"
        subtitle = f"<span style='font-size: 12px;'>Geral</span>"
        title = f"{main_title}"

        df_empresas_agrupadas = df_filtrado_empresas_graph.groupby(['data'], as_index=False)['valor'].sum()
        
        fig_box = px.box(
            df_empresas_agrupadas, 
            y='valor', 
            points="all",
            hover_data={'data': '|%d/%m/%y', 'valor':':,.2f'},
            title=title,    
            height=391.5,
            color_discrete_sequence=  [util.barra_azul],
        )

        fig_box.update_layout(
            margin=dict(l=0, r=0, t=45, b=0),
            title_font_color="rgb(98,83,119)",
            title_font_size=15,
            showlegend=False,
        )

        fig_box.update_yaxes(title_text=f"Faturado ({mes_nome_selecionado}/{ano_selecionado})", 
                                showline=False, 
                                linecolor="Grey",
                                linewidth=0.1, 
                                gridcolor='lightgrey',
                                autorange='max reversed',
                                dtick=200,
                                zerolinecolor='lightgrey',
                                showticklabels=True,
        )

        fig_box.update_xaxes( 
                                showline=True, 
                                linecolor="Grey", 
                                linewidth=0.1, 
                                gridcolor='lightgrey',
                                title_text=f'{x_title}'
        )
        fig_box.update_traces(marker=dict(size=4.5),
                                boxmean='sd',)

        col9.plotly_chart(fig_box, use_container_width=True, automargin=True)

    except pd.errors.ParserError as e:
        print(f"Erro ao analisar o arquivo CSV: {e}")
    except Exception as e:
        col9.warning(f"Ocorreu um erro: {e}")

########################################################################################
####### GRAFICO BARRAS DISTRIBUIÇÃO FATURAMENTO MENSAL POR SERVIÇO #####################
########################################################################################

try:
    main_title = f"-DISTRIBUIÇÃO FATURAMENTO ({mes_nome_selecionado.upper()} DE {ano_selecionado})"
    empresa_part = f"Empresa: {empresa_nome}" if empresa_nome != "Todas" else "Todas as Empresas"
    subtitle = f"<span style='font-size: 12px;'>{empresa_part}</span>"
    title = f"{main_title} | {subtitle}"

    df_proporcao_por_empresa = df_filtrado_empresas_graph.groupby('empresa')['valor'].sum().reset_index()
    df_proporcao_por_empresa['porcentagem'] = (df_proporcao_por_empresa['valor'] / df_proporcao_por_empresa['valor'].sum()) * 100
    df_proporcao_por_empresa['porcentagem_str'] = df_proporcao_por_empresa['porcentagem'].apply(lambda x: f"{x:.2f}%")
    df_proporcao_por_empresa = df_proporcao_por_empresa.sort_values(by='valor', ascending=False).reset_index()
    
    df_proporcao_por_servico = df_filtrado_empresas_graph.groupby('servico')['valor'].sum().reset_index()
    df_proporcao_por_servico['porcentagem'] = (df_proporcao_por_servico['valor'] / df_proporcao_por_servico['valor'].sum()) * 100
    df_proporcao_por_servico['porcentagem_str'] = df_proporcao_por_servico['porcentagem'].apply(lambda x: f"{x:.2f}%")
    df_proporcao_por_servico = df_proporcao_por_servico.sort_values(by='valor', ascending=False).reset_index()

    colors = [util.barra_verde if i % 2 == 0 else util.barra_azul for i in range(len(df_proporcao_por_servico))]

    fig_servico = go.Figure()

    if empresa_nome == 'Todas':
        for i, row in df_proporcao_por_empresa.iterrows():
            fig_servico.add_trace(go.Bar(
                x = [row['empresa']],
                y=[row['porcentagem']],
                text=row['porcentagem_str'],
                textposition='inside' if row['porcentagem'] > 10 else 'outside',
                marker_color=colors[i],
                insidetextfont=dict(size=14, color="white"),
                outsidetextfont=dict(size=14, color=util.barra_azul_escuro)
            ))
        fig_servico.update_yaxes(range=[0, df_proporcao_por_empresa['porcentagem'].max() + 10])

    else:
        for i, row in df_proporcao_por_servico.iterrows():
            fig_servico.add_trace(go.Bar(
                x = [row['servico']],
                y=[row['porcentagem']],
                text=row['porcentagem_str'],
                textposition='inside' if row['porcentagem'] > 10 else 'outside',
                marker_color=colors[i],
                insidetextfont=dict(size=14, color="white"),
                outsidetextfont=dict(size=14, color=util.barra_azul_escuro)
        ))
        fig_servico.update_yaxes(range=[0, df_proporcao_por_servico['porcentagem'].max() + 10])

    fig_servico.update_layout(
        title=main_title,
        showlegend=False,
        height=391.5,
        margin=dict(l=0, r=0, t=45, b=0),
        title_font_color="rgb(98,83,119)",
        title_font_size=15,
        uniformtext_minsize=14,
        uniformtext_mode='hide',
        barmode='group'

    )

    fig_servico.update_yaxes(
        showline=True, 
        linecolor="Grey", 
        linewidth=0.1, 
        gridcolor='lightgrey', 
        title="Porcentagem", 
        dtick=5, 
    )

    fig_servico.update_xaxes(
        showline=True, 
        linecolor="Grey", 
        linewidth=0.1, 
        gridcolor='lightgrey', 
        title_text=f'{x_title}'
    )
    
    col8.plotly_chart(fig_servico, use_container_width=True, automargin=True)
except pd.errors.ParserError as e:
    print(f"Erro ao analisar o arquivo CSV: {e}")
except Exception as e:
    col8.warning(f"Ocorreu um erro: {e}")


########################################################################################
####### GRAFICO CONTROLE DE CAPITAL ####################################################
########################################################################################

df_capital = df_capital[df_capital['categoria'] == 'Capital']

df_capital = df_capital[df_capital['empresa'] == empresa_nome]

grouped_diario = df_capital.groupby([pd.Grouper(key='data', freq='D'), 'tipo']).sum(numeric_only=True).reset_index()
grouped_diario['valor_formatado'] = grouped_diario['valor'].apply(util.formata_para_brl)

total_por_dia = grouped_diario.groupby('data')['valor'].sum().reset_index()
total_por_dia['valor_formatado'] = total_por_dia['valor'].apply(util.formata_para_brl)
total_por_dia['data_formatada'] = total_por_dia['data'].dt.strftime('%d/%m')

primeiro_valor = total_por_dia['valor'].iloc[0]
ultimo_valor = total_por_dia['valor'].iloc[-1]
if primeiro_valor != 0:
    diferenca_percentual = ((ultimo_valor - primeiro_valor) / primeiro_valor) * 100
else:
    diferenca_percentual = 0 if ultimo_valor == 0 else float('inf')
diferenca_formatada = f"{diferenca_percentual:.2f}%"
if diferenca_percentual >= 0:
    diferenca_formatada = f"+{diferenca_formatada}"

penultimo_valor = total_por_dia['valor'].iloc[-2]
if penultimo_valor != 0:
    diferenca_percentual_ultimo_anterior = ((ultimo_valor - penultimo_valor) / penultimo_valor) * 100
else:
    diferenca_percentual_ultimo_anterior = 0 if ultimo_valor == 0 else float('inf')
diferenca_formatada_ultimo_anterior = f"{diferenca_percentual_ultimo_anterior:.2f}%"
if diferenca_percentual_ultimo_anterior >= 0:
    diferenca_formatada_ultimo_anterior = f"+{diferenca_formatada_ultimo_anterior}"

valor_quatro_semanas = total_por_dia['valor'].iloc[-4]
if valor_quatro_semanas != 0:
    diferenca_percentual_quatro_semanas = ((ultimo_valor - valor_quatro_semanas) / valor_quatro_semanas) * 100
else:
    diferenca_percentual_quatro_semanas = 0 if ultimo_valor == 0 else float('inf')
diferenca_formatada_quatro_semanas = f"{diferenca_percentual_quatro_semanas:.2f}%"
if diferenca_percentual_quatro_semanas >= 0:
    diferenca_formatada_quatro_semanas = f"+{diferenca_formatada_quatro_semanas}"

cor_anotacao = util.barra_azul if diferenca_percentual >= 0 else util.barra_vermelha
cor_anotacao_ultimo_anterior = util.barra_azul if diferenca_percentual_ultimo_anterior >= 0 else util.barra_vermelha
cor_anotacao_quatro_semanas = util.barra_azul if diferenca_percentual_quatro_semanas >= 0 else util.barra_vermelha

grouped_diario['data'] = pd.to_datetime(grouped_diario['data'])
ultima_data = grouped_diario['data'].max()
df_ultima_data = grouped_diario[grouped_diario['data'] == ultima_data]
valor_a_receber = df_ultima_data[df_ultima_data['tipo'] == 'A Receber']['valor_formatado'].values[0]
em_caixa = df_ultima_data[df_ultima_data['tipo'] == 'Disponível']['valor_formatado'].values[0]

grafico_capital = px.bar(
    grouped_diario, 
    x='data', 
    y='valor',
    color='tipo',
    orientation='v',
    labels={'tipo': '', 'data': 'Data', 'valor': 'Capital'},  
    color_discrete_sequence=[util.barra_azul, util.barra_verde]  
)

for i in range(len(grafico_capital.data)):
    grafico_capital.data[i].text = grouped_diario[grouped_diario['tipo'] == grafico_capital.data[i].name]['valor_formatado']
    grafico_capital.data[i].texttemplate = '%{text}'
    grafico_capital.data[i].textposition = 'inside'

grafico_capital.add_trace(
    go.Scatter(
        x=total_por_dia['data'], 
        y=total_por_dia['valor'], 
        name='Total Capital',
        mode='lines+markers',
        line=dict(color=util.barra_vermelha),
        marker=dict(color=util.barra_vermelha),
        text=total_por_dia['valor_formatado'],
        textposition='top center',
        texttemplate='%{text}'
    )
)

main_title = f"- CONTROLE DE CAPITAL"
subtitle = f"<span style='font-size: 12px;'>Todas as Empresas</span>"
title = f"{main_title} | {subtitle}"

grafico_capital.update_layout(
    margin=dict(l=0, r=0, t=45, b=0),
    height=391.5,
    title=title,
    xaxis_title="",
    yaxis_title="Capital",
    font=dict(size=13, color='rgb(249,250,255)'),

    title_font_color="rgb(98,83,119)",
    title_font_size=15,

    legend=dict(x=0, y=1.01, orientation='h'),
    annotations=[
        go.layout.Annotation(
            text=f"<b>Disponível</b>: {em_caixa}",
            xref="paper",
            yref="paper",
            x=1,
            y=1.13,
            showarrow=False,
            font=dict(size=14, color=util.barra_azul)
        ),
        go.layout.Annotation(
            text=f"<b>A Receber:</b> {valor_a_receber}",
            xref="paper",
            yref="paper",
            x=1,
            y=1.06,
            showarrow=False,
            font=dict(size=14, color=util.barra_azul)
        )
    ],
)


grafico_capital.update_yaxes(showline=True, linecolor="Grey", linewidth=0.1, gridcolor='lightgrey', dtick=1000, range=[0, total_por_dia['valor'].max() + 10000], showticklabels=False)
grafico_capital.update_xaxes(showline=True, linecolor="Grey", linewidth=0.1, gridcolor='lightgrey', title="Período")

grafico_capital.update_xaxes(
    tickmode='array', 
    tickvals=total_por_dia['data'], 
    ticktext=total_por_dia['data_formatada']
)

c200.plotly_chart(grafico_capital, use_container_width=True, automargin=True)

with c2000:

    col205 = st.container()
    col204 = st.container()
    
    col201 = st.container()
    col202 = st.container()
    col203 = st.container()

    col205.success(f"{util.formata_para_brl(ultimo_valor)}", icon=":material/account_balance:")

    if (ultimo_valor - primeiro_valor) >= 0:
            col204.success(f"{util.formata_para_brl(ultimo_valor - primeiro_valor)}", icon=":material/add:")
    else:
            col204.error(f"{util.formata_para_brl(ultimo_valor - primeiro_valor)}", icon=":material/remove:")

    if diferenca_percentual_ultimo_anterior >= 0:
        col201.info(f"{diferenca_formatada_ultimo_anterior}", icon=":material/replay_10:")
    else:
        col201.error(f"{diferenca_formatada_ultimo_anterior}", icon=":material/replay_10:")

    if diferenca_percentual_quatro_semanas >= 0:
        col202.info(f"{diferenca_formatada_quatro_semanas}", icon=":material/replay_30:")
    else:
        col202.error(f"{diferenca_formatada_quatro_semanas}", icon=":material/replay_30:")

    if diferenca_percentual >= 0:
        col203.info(f"{diferenca_formatada}", icon=":material/all_inclusive:")
    else:
        col203.error(f"{diferenca_formatada}", icon=":material/all_inclusive:")


########################################################################################
####### ABA COMPORTAMENTO MENSAL #######################################################
########################################################################################
with tab_objects[1]:
    with st.container(border=True):
        col9, col10, col11, col304 = st.columns([1,2,2,1])
        
        border_c = st.container(border=True)
        with border_c:
            col12 , col300= st.columns([3,1])

    ########################################################################################
    ####### GRAFICO BARRA FATURAMENTO MENSAL ###############################################
    ########################################################################################
    empresas = sorted(df_faturamento['empresa'].unique())
    empresa_nome = col10.selectbox('Selecione a Empresa:', empresas, key = "empresas_mensal", disabled=True)
    try:

        df_faturamento = df_faturamento[df_faturamento['empresa'] == empresa_nome]

        anos_disponiveis = util.anos_disponiveis(df_faturamento)

        ano_selecionado = col9.selectbox('Selecione o Ano:', anos_disponiveis, index=len(anos_disponiveis)-1)

        if empresa_nome == 'Todos':
            servicos = ['Todos'] + sorted(df_faturamento['servico'].unique())
        else:
            servicos = ['Todos'] + sorted(df_faturamento[df_faturamento['empresa'] == empresa_nome]['servico'].unique())
        selected_servico = col11.selectbox('Selecione o Serviço:', servicos, key = "servico_mensal")

        df_filtrado_empresas = df_faturamento.copy()
        if empresa_nome != 'Todas':
            df_filtrado_empresas = df_filtrado_empresas[df_filtrado_empresas['empresa'] == empresa_nome]
        if selected_servico != 'Todos':
            df_filtrado_empresas = df_filtrado_empresas[df_filtrado_empresas['servico'] == selected_servico]

        df_filtered_mensal = df_filtrado_empresas[df_filtrado_empresas['data'].dt.year == ano_selecionado]

        grouped_mensal = df_filtered_mensal.groupby(df_filtered_mensal['data'].dt.month).sum(numeric_only=True).reset_index()
        grouped_mensal['mês'] = grouped_mensal['data'].apply(lambda x: f"{util.mapa_meses[x]}/{ano_selecionado}")

        data_final_disponivel = df_filtered_mensal['data'].max()
        
        ultimo_mes = data_final_disponivel.month
        dias_no_mes = data_final_disponivel.days_in_month
        dias_ate_data_final = data_final_disponivel.day

        dados_ultimo_mes = df_filtered_mensal[df_filtered_mensal['data'].dt.month == ultimo_mes]
        total_ultimo_mes = dados_ultimo_mes['valor'].sum(numeric_only=True)

        valor_previsao = (total_ultimo_mes / dias_ate_data_final) * dias_no_mes

        grouped_mensal.loc[grouped_mensal['data'] == ultimo_mes, 'previsao'] = valor_previsao
        grouped_mensal['previsao'] = grouped_mensal['previsao'].fillna(0)

        grouped_mensal = grouped_mensal.rename(columns={"valor": "Faturado", "previsao": "Previsto"})
        
        grafico_faturamento_barra = px.bar(
            grouped_mensal, 
            x='mês', 
            y=['Previsto', 'Faturado'], 
            title='Valor Real e Previsão por Mês',
            color_discrete_sequence=["#c6ccd2",util.barra_azul]
        )

        grafico_faturamento_barra.update_traces(textposition='outside')
        
        grafico_faturamento_barra.data[0].text = [
            f'Previsto: {util.formata_para_brl(valor)}' for valor in grouped_mensal['Previsto']
        ]

        main_title = f"-FATURAMENTO MENSAL ({ano_selecionado})"
        empresa_part = f"Empresa: {empresa_nome}" if empresa_nome != "Todas" else "Todas as Empresas"
        servico_part = f"({selected_servico})" if selected_servico != "Todos" else ""
        subtitle = f"<span style='font-size: 12px;'>{empresa_part} {servico_part}</span>"
        title = f"{main_title} | {subtitle}"

        grafico_faturamento_linha = px.line(
            grouped_mensal, 
            x='mês', 
            y='Faturado',
            text=grouped_mensal['Faturado'].apply(lambda x: f"R$ {x:,.2f}".replace('.', '|').replace(',', '.').replace('|', ',')),
            color_discrete_sequence=[util.barra_vermelha],
            title=title,
            labels={'mês': 'Período', 'Faturado': 'Faturado'}
        )
        grafico_faturamento_linha.update_traces(textposition='top center')
        grafico_faturamento_barra.update_traces(textposition='outside')


        combined_fig = go.Figure()

        for trace in grafico_faturamento_linha.data:
            combined_fig.add_trace(trace)

        for trace in grafico_faturamento_barra.data:
            combined_fig.add_trace(trace)

        combined_fig.update_layout(
            title='Gráfico Combinado',
            xaxis_title='Período',
            yaxis_title='Valores',
            yaxis2=dict(
                title='População',
                overlaying='y',
                side='right'
            )
        )

        combined_fig.update_yaxes(showline=True, linecolor="Grey", linewidth=0.1, gridcolor='lightgrey', dtick=5000, range=[0, grouped_mensal['Faturado'].max() * 1.3])
        combined_fig.update_xaxes(showline=True, linecolor="Grey", linewidth=0.1, gridcolor='lightgrey')
        combined_fig.update_layout(
            title=title, 
            margin=dict(l=0, r=0, t=45, b=0), 
            height=366, title_font_color="rgb(98,83,119)", 
            font=dict(size=13, color='#16181c'),
            barmode='overlay',
            xaxis_title='Período',
            yaxis_title="Receita",
            legend_title_text='Análise Preditiva',
            showlegend=False,

        )

        col12.plotly_chart(combined_fig, use_container_width=True, automargin=True)
   
    except pd.errors.ParserError as e:
        print(f"Erro ao analisar o arquivo CSV: {e}")
    except Exception as e:
        col12.warning(f"Ocorreu um erro: {e}")

    ########################################################################################
    ####### GRÁFICO BOX FATURAMENTO MENSAL #################################################
    ########################################################################################    
    df_mensal_box = df_faturamento.copy()

    if empresa_nome != 'Todas':
        df_mensal_box = df_mensal_box[df_mensal_box['empresa'] == empresa_nome]
    if selected_servico != 'Todos':
        df_mensal_box = df_mensal_box[df_mensal_box['servico'] == selected_servico]

    main_title = f'-BOX PLOT FATURAMENTO MENSAL ({ano_selecionado})'
    subtitle = f"<span style='font-size: 12px;'>{empresa_part} {servico_part}</span>"
    title=f"{main_title} | {subtitle}"

    grouped_mensal['ano'] = ano_selecionado

    try:
        fig_box_mensal = px.box(
            grouped_mensal, 
            x='ano', 
            y= np.where(grouped_mensal['Previsto'] == 0, grouped_mensal['Faturado'], grouped_mensal['Previsto']),
            color='ano',
            points="all",
            title=" ",    
            height=366,
            color_discrete_sequence= ["#2d5480"],
            hover_data={'mês'}
        )
        
        fig_box_mensal.update_layout(
            margin=dict(l=0, r=0, t=45, b=0),
            title_font_color="rgb(98,83,119)",
            showlegend=False,
        )
        fig_box_mensal.update_yaxes(title_text=f"", 
                                showline=False, 
                                linecolor="Grey",
                                linewidth=0.1, 
                                gridcolor='lightgrey',
                                autorange=True,
                                dtick=5000,
                                zerolinecolor='lightgrey',
                                showticklabels=True,
        )

        fig_box_mensal.update_xaxes( 
                                showline=True, 
                                linecolor="Grey", 
                                linewidth=0.1, 
                                gridcolor='lightgrey',
                                title_text='Período'
        )
        fig_box_mensal.update_traces(marker=dict(size=4.5),
                                boxmean='sd',)
        col300.plotly_chart(fig_box_mensal, use_container_width=True, automargin=True)

    except pd.errors.ParserError as e:
        print(f"Erro ao analisar o arquivo CSV: {e}")
    except Exception as e:
        col300.warning(f"Ocorreu um erro: {e}")


    ########################################################################################
    ####### TABELA FATURAMENTO MENSAL ##########################################################
    ########################################################################################
    
    meses_disponiveis = sorted(df_filtered_mensal['data'].dt.month.unique())
    mes_nomes_disponiveis = [util.mapa_meses[mes] for mes in meses_disponiveis]
    mes_selecionado = col304.selectbox('Filtro Mensal:', mes_nomes_disponiveis, index=len(mes_nomes_disponiveis)-1, key="mes_faturamento")
    mes_selecionado_num = list(util.mapa_meses.keys())[list(util.mapa_meses.values()).index(mes_selecionado)]
    
    df_tabela_filtrado = df_filtered_mensal[df_filtered_mensal['data'].dt.month == mes_selecionado_num]
    
    df_tabela_agrupado = df_tabela_filtrado.groupby('servico').agg({
        'data': 'first',    
        'empresa': 'first',
        'valor': 'sum'     
    }).reset_index()

    df_tabela_agrupado = df_tabela_agrupado[['data', 'empresa', 'servico', 'valor']]    
    
    df_tabela_agrupado = df_tabela_agrupado.fillna(0)

    df_tabela_agrupado['data'] = df_tabela_agrupado['data'].dt.month.apply(lambda x: f"{util.mapa_meses[x]}/{ano_selecionado}")

    df_tabela_agrupado = df_tabela_agrupado.sort_values(by=['empresa', 'valor'], ascending=[True, False])

    df_tabela_agrupado['valor'] = df_tabela_agrupado['valor'].apply(util.formata_para_brl)

    df_tabela_agrupado = df_tabela_agrupado.rename(columns={
        'data': 'Data',
        'empresa': 'Empresa',
        'servico': 'Serviço',
        'valor': 'Total Faturado'
    })
    
    def style_alternate_rows(row):
        styles = [''] * len(row)
        index_in_sorted = df_tabela_agrupado.index.get_loc(row.name)
        if index_in_sorted % 2 == 0:  
            styles = ['background-color: #245682; font-weight: bold; color: white' if col in ['Data', 'Total Faturado'] else 'background-color: #dddddd; color: #16181c' for col in row.index]
        else:
            styles = ['background-color: #3e709b; font-weight: bold; color: white' if col in ['Data', 'Total Faturado'] else 'background-color: #eeeeee; color: #16181c' for col in row.index]
        return styles

    df_style_despesa = df_tabela_agrupado.style.apply(style_alternate_rows, axis=1)
    
    base_height = 37  
    row_height = 35    

    height = base_height + len(df_tabela_agrupado) * row_height

    with border_c:

        col302 = st.container()
        col302.dataframe(df_style_despesa, height=height, use_container_width=True, hide_index=True)


########################################################################################
####### ABA DESPESAS #######################################################
########################################################################################
with tab_objects[2]:
    with st.container(border=True):
        col14, col16, col17, col18, col404 = st.columns([1, 1, 1, 1, 1])

        border_c2 = st.container(border=True)
        with border_c2:
            c4, c401 = st.columns([3, 1])

            checkbox_despsa = st.container(border=True)

    ########################################################################################
    ####### GRÁFICO BARRAS DESPESA MENSAL ##################################################
    ########################################################################################
    empresas = sorted(df_despesa['empresa'].unique())
    empresa_selecionada = col16.selectbox('Selecione a Empresa:', empresas, index=0, key="empresa_despesa", disabled=True)

    df_despesa = df_despesa[df_despesa['empresa'] == empresa_selecionada]

    try:
        df_despesa['empresa'] = df_despesa['empresa'].astype(str)
        df_despesa['tipo'] = df_despesa['tipo'].astype(str)
        df_despesa['descricao'] = df_despesa['descricao'].astype(str)

        anos_disponiveis = util.anos_disponiveis(df_despesa)   
        ano_selecionado = col14.selectbox('Selecione o Ano:', anos_disponiveis, index=len(anos_disponiveis)-1, key="ano_despesa")

        def atualizar_filtros(ano, empresa=None, tipo=None, descricao=None):
            df_filtrado = df_despesa[df_despesa['data'].dt.year == ano]
            if empresa and empresa != "Todas":
                df_filtrado = df_filtrado[df_filtrado['empresa'] == empresa]
            if tipo and tipo != "Todos":
                df_filtrado = df_filtrado[df_filtrado['tipo'] == tipo]
            if descricao and descricao != "Todas":
                df_filtrado = df_filtrado[df_filtrado['descricao'] == descricao]
            return df_filtrado

        df_filtrado = atualizar_filtros(ano_selecionado)
        empresas = sorted(list(df_filtrado['empresa'].unique()))
        tipos = ["Todos"] + sorted(list(df_filtrado['tipo'].unique()))
        descricoes = ["Todas"] + sorted(list(df_filtrado['descricao'].unique()))

        df_filtrado = atualizar_filtros(ano_selecionado, empresa=empresa_selecionada)
        tipos = ["Todos"] + sorted(list(df_filtrado['tipo'].unique()))
        descricoes = ["Todas"] + sorted(list(df_filtrado['descricao'].unique()))

        tipo_selecionado = col17.selectbox('Selecione o Tipo de Despesa:', tipos, index=0, key="tipo_despesa")
        df_filtrado = atualizar_filtros(ano_selecionado, empresa=empresa_selecionada, tipo=tipo_selecionado)
        descricoes = ["Todas"] + sorted(list(df_filtrado['descricao'].unique()))

        descricao_selecionada = col18.selectbox('Selecione a Despesa:', descricoes, index=0, key="descricao_despesa")
        df_filtrado = atualizar_filtros(ano_selecionado, empresa=empresa_selecionada, tipo=tipo_selecionado, descricao=descricao_selecionada)

        soma_por_mes = df_filtrado.groupby(df_filtrado['data'].dt.month)['valor'].sum(numeric_only=True).reset_index()
        soma_por_mes['mês'] = soma_por_mes['data'].apply(lambda x: f"{util.mapa_meses[x]}/{ano_selecionado}")

        main_title = f'-DESPESA MENSAL ({ano_selecionado})'
        empresa_part = f"Empresa: {empresa_selecionada}" if empresa_selecionada != "Todas" else "Todas as Empresas"
        tipo_desp = f"({tipo_selecionado})" if tipo_selecionado != "Todos" else ""
        subtitle = f"<span style='font-size: 12px;'>{empresa_part} {tipo_desp}</span>"
        title = f"{main_title} | {subtitle}"

        grafico_despesa_barra = px.bar(
            soma_por_mes, 
            x='mês', 
            y='valor',
            color_discrete_sequence=[util.barra_vermelha],
            title=title,
            labels={'mês': 'Período', 'valor': 'Despesa'}
        )

        grafico_despesa_linha = px.line(
            soma_por_mes, 
            x='mês', 
            y='valor',
            text=soma_por_mes['valor'].apply(lambda x: f"R$ {x:,.2f}".replace('.', '|').replace(',', '.').replace('|', ',')),
            color_discrete_sequence=["#21080b"],
            title=title,
            labels={'mês': 'Período', 'valor': 'Despesa'}
        )
        grafico_despesa_linha.update_traces(textposition='top center')

        combined_fig = go.Figure()

        for trace in grafico_despesa_linha.data:
            combined_fig.add_trace(trace)

        for trace in grafico_despesa_barra.data:
            combined_fig.add_trace(trace)

        combined_fig.update_layout(
            title='Gráfico Combinado',
            xaxis_title='Período',
            yaxis_title='Valores',
            yaxis2=dict(
                title='População',
                overlaying='y',
                side='right'
            )
        )

        combined_fig.update_yaxes(showline=True, linecolor="Grey", linewidth=0.1, gridcolor='lightgrey', dtick=5000, range=[0, grouped_mensal['Faturado'].max() * 1.3])
        combined_fig.update_xaxes(showline=True, linecolor="Grey", linewidth=0.1, gridcolor='lightgrey')
        combined_fig.update_layout(title=title, margin=dict(l=0, r=0, t=45, b=0), height=366, title_font_color="rgb(98,83,119)", font=dict(size=14, color='#16181c'))

        c4.plotly_chart(combined_fig, use_container_width=True, automargin=True)


        ########################################################################################
        ####### BOX PLOT DESPESA MENSAL ##########################################################
        ########################################################################################
        soma_por_mes['ano'] = ano_selecionado

        fig_box_mensal = px.box(
            soma_por_mes, 
            x='ano', 
            y='valor', 
            color='ano',
            points="all",
            title=" ",    
            height=366,
            color_discrete_sequence=  ["#a22938"],
            hover_data={'mês'},
        )

        fig_box_mensal.update_layout(
            margin=dict(l=0, r=0, t=45, b=0),
            title_font_color="rgb(98,83,119)",
            showlegend=False,
        )
        fig_box_mensal.update_yaxes(title_text=f"", 
                                showline=False, 
                                linecolor="Grey",
                                linewidth=0.1, 
                                gridcolor='lightgrey',
                                dtick=5000,
                                zerolinecolor='lightgrey',
                                showticklabels=True,
        )

        fig_box_mensal.update_xaxes( 
                                showline=True, 
                                linecolor="Grey", 
                                linewidth=0.1, 
                                gridcolor='lightgrey',
                                title_text='Período'
        )
        fig_box_mensal.update_traces(marker=dict(size=4.5),
                                boxmean='sd',)

        c401.plotly_chart(fig_box_mensal, use_container_width=True, automargin=True)


        ########################################################################################
        ####### TABELA DESPESA MENSAL ##########################################################
        ########################################################################################

        meses_disponiveis = sorted(df_filtrado['data'].dt.month.unique())
        mes_nomes_disponiveis = [util.mapa_meses[mes] for mes in meses_disponiveis]
        mes_selecionado = col404.selectbox('Filtro Mensal:', mes_nomes_disponiveis, index=len(mes_nomes_disponiveis)-1, key="mes_despesa")
        mes_selecionado_num = list(util.mapa_meses.keys())[list(util.mapa_meses.values()).index(mes_selecionado)]

        filtrar_valor_maior_que_zero = checkbox_despsa.toggle('Apenas Despesas Lançadas', value=True)

        df_tabela_filtrado = df_filtrado[df_filtrado['data'].dt.month == mes_selecionado_num]

        if filtrar_valor_maior_que_zero:
            df_tabela_filtrado = df_tabela_filtrado[df_tabela_filtrado['valor'] > 0]

        df_tabela_filtrado = df_tabela_filtrado.fillna(0)

        df_tabela_filtrado['data'] = df_tabela_filtrado['data'].dt.month.apply(lambda x: f"{util.mapa_meses[x]}/{ano_selecionado}")

        df_tabela_filtrado['valor'] = df_tabela_filtrado['valor'].apply(util.formata_para_brl)

        df_tabela_filtrado = df_tabela_filtrado.rename(columns={
            'data': 'Data',
            'empresa': 'Empresa',
            'tipo': 'Tipo de Despesa',
            'descricao': 'Descrição',
            'valor': 'Valor'
        })

        df_tabela_filtrado = df_tabela_filtrado.sort_values(by=['Empresa', 'Tipo de Despesa', 'Descrição'])

        df_tabela_filtrado = df_tabela_filtrado.drop(columns=['categoria'])

        def style_alternate_rows(row):
            styles = [''] * len(row)
            index_in_sorted = df_tabela_filtrado.index.get_loc(row.name)
            if index_in_sorted % 2 == 0:  
                styles = ['background-color: #a22938; font-weight: bold; color: white' if col in ['Data', 'Valor'] else 'background-color: #dddddd; color: #16181c' for col in row.index]
            else:
                styles = ['background-color: #b85061; font-weight: bold; color: white' if col in ['Data', 'Valor'] else 'background-color: #eeeeee; color: #16181c' for col in row.index]
            return styles

        df_style_despesa = df_tabela_filtrado.style.apply(style_alternate_rows, axis=1)

        base_height = 37  
        row_height = 35    

        height = base_height + len(df_tabela_filtrado) * row_height

        with border_c2:
            c400 = st.container()
            c400.dataframe(df_style_despesa, height=height, use_container_width=True, hide_index=True)

    except pd.errors.ParserError as e:
        print(f"Erro ao analisar o arquivo CSV: {e}")
    except Exception as e:
        c4.warning(f"Ocorreu um erro: {e}")


########################################################################################
####### ABA ANÁLISE | BALANCETE GERAL ##################################################
########################################################################################
with tab_objects[3]:
    with st.container(border=True):
        c20,c21 = st.columns([2,2])
        with st.container(border=True):
            c2 = st.container()

    with st.expander("INDICADORES: CUSTOS / RECEITA TOTAL", expanded=False, icon=":material/query_stats:"):
        with st.container(border=False):
            c10000,c10100 = st.columns([2,2])
            with c10000:
                c100 = st.container(border=True)
                c102 = st.container(border=True)
            with c10100:
                c101 = st.container(border=True)
                c103 = st.container(border=True)

 
    with st.expander("DRE: DEMOSNTRATIVO DE RESULTADOS DO EXERCÍCIO", expanded=False,  icon=":material/finance_mode:"):
        with st.container(border=True):
            c104, c105 = st.columns([2,2])



    empresas = sorted(df_faturamento['empresa'].unique())


    empresa_selecionada = c21.selectbox('Selecione a Empresa:', empresas, key = "empresas_cmv", index=0, disabled=True)

    df_despesa = df_despesa[df_despesa['empresa'] == empresa_selecionada]
 
    anos_disponiveis = sorted(df_despesa['data'].dt.year.unique())
    ano_selecionado = c20.selectbox('Selecione o Ano:', anos_disponiveis, index=len(anos_disponiveis)-1, key="ano_faturameto_despesa")

    #############################################################################
    ####### GRÁFICO CMV #########################################################
    #############################################################################
    if empresa_selecionada == "Todas":
        df_faturamento_cmv = df_faturamento[(df_faturamento['data'].dt.year == ano_selecionado) & (df_faturamento['cmv'] == 1)]
        df_despesa_cmv = df_despesa[(df_despesa['data'].dt.year == ano_selecionado) & (df_despesa['tipo'].str.contains("CMV"))]

        df_faturamento_cmv_filtrado = df_faturamento_cmv.groupby(df_faturamento_cmv['data'].dt.month)['valor'].sum(numeric_only=True).reset_index()
        df_despesa_cmv_filtrado = df_despesa_cmv.groupby(df_despesa_cmv['data'].dt.month)['valor'].sum(numeric_only=True).reset_index()

        merged_data_filtered_cmv = pd.merge(df_faturamento_cmv_filtrado, df_despesa_cmv_filtrado, on='data', how='left', suffixes=('_faturamento', '_despesa'))
        merged_data_filtered_cmv = merged_data_filtered_cmv.fillna(0)
    else:
        if empresa_selecionada == "Pousada da Ponte":
            df_faturamento_cmv = df_faturamento[(df_faturamento['data'].dt.year == ano_selecionado) & (df_faturamento['empresa'].str.contains(f"{empresa_selecionada}")) & (df_faturamento['servico'].str.contains("Frigobar"))]
        else:
            df_faturamento_cmv = df_faturamento[(df_faturamento['data'].dt.year == ano_selecionado) & df_faturamento['empresa'].str.contains(f"{empresa_selecionada}")]
        df_despesa_cmv = df_despesa[(df_despesa['data'].dt.year == ano_selecionado) & (df_despesa['empresa'].str.contains(f"{empresa_selecionada}")) & (df_despesa['tipo'].str.contains("CMV"))]

        df_faturamento_cmv_filtrado = df_faturamento_cmv.groupby(df_faturamento_cmv['data'].dt.month)['valor'].sum(numeric_only=True).reset_index()
        df_despesa_cmv_filtrado = df_despesa_cmv.groupby(df_despesa_cmv['data'].dt.month)['valor'].sum(numeric_only=True).reset_index()

        merged_data_filtered_cmv = pd.merge(df_faturamento_cmv_filtrado, df_despesa_cmv_filtrado, on='data', how='left', suffixes=('_faturamento', '_despesa'))
        merged_data_filtered_cmv = merged_data_filtered_cmv.fillna(0)

    merged_data_filtered_cmv['percentage_cmv'] = (merged_data_filtered_cmv['valor_despesa'] / merged_data_filtered_cmv['valor_faturamento']) * 100
    merged_data_filtered_cmv['percentage_cmv'] = merged_data_filtered_cmv['percentage_cmv'].fillna(0)
    merged_data_filtered_cmv['percentage_cmv_str'] = merged_data_filtered_cmv['percentage_cmv'].apply(lambda x: f"{x:.2f}%")
    merged_data_filtered_cmv['mês'] = merged_data_filtered_cmv['data'].apply(lambda x: f"{util.mapa_meses[x]}/{ano_selecionado}")

    grafico_bar_faturamento = go.Bar(
        x=merged_data_filtered_cmv['mês'], 
        y=merged_data_filtered_cmv['valor_faturamento'], 
        name="Receita Total Sob Consumo",
        marker_color=util.barra_cinza_escuro,
        yaxis='y1'
    )

    grafico_cmv = go.Scatter(
        x=merged_data_filtered_cmv['mês'],
        y=merged_data_filtered_cmv['percentage_cmv'],
        mode='lines+markers+text',
        name='CMV / Receita Total',
        text=merged_data_filtered_cmv['percentage_cmv_str'],
        textposition='top center',
        marker=dict(color=util.barra_vermelha),
        yaxis='y2'  
    )

    fig_cmv = go.Figure(data=[grafico_bar_faturamento, grafico_cmv])

    main_title = f'-CMV / RECEITA TOTAL SOB CONSUMO ({ano_selecionado})'
    empresa_part = f"{empresa_selecionada}" if empresa_selecionada != "Todas" else "Todas as Empresas"
    subtitle = f"<span style='font-size: 12px;'>{empresa_part}</span>"
    title = f"{main_title} | {subtitle}"
    margem = 90   

    fig_cmv.update_layout(
        title=title,
        title_font_color="rgb(98,83,119)",
        title_font_size=15,
        xaxis=dict(
            title="",
            mirror=True,
            showline=True,
            linecolor="Grey",
            linewidth=0.5,
            gridcolor='lightgrey',
            ticks='outside'
        ),
        yaxis=dict(
            title='Receita Total',
            mirror=True,
            showline=True,
            linecolor="Grey",
            linewidth=0.5,
            gridcolor='lightgrey',
            ticks='outside',
            range=[0, merged_data_filtered_cmv['valor_faturamento'].max() * 1.2], 
            dtick=10000  
        ),
        yaxis2=dict(
            title='CMV (%)',
            overlaying='y',
            side='right',
            range=[0, 100],  
            tickvals=[0, 20, 40, 60, 80, 100], 
            tickformat='.0f', 
            showline=True,
            linecolor="Grey",
            linewidth=0.5,
            ticks='outside',
            gridcolor='lightgrey',
            showgrid=False
        ),
        legend=dict(x=0.1, y=1.1, orientation='h'),
        margin=dict(l=0, r=0, t=70, b=0),
        height=320,
        font=dict(size=14, color='rgb(98,83,119)'),
    )

    c100.plotly_chart(fig_cmv, use_container_width=True, automargin=True)


    ########################################################################################
    ####### GRÁFICO DESPESA COM PESSOAL ####################################################
    ########################################################################################
    if empresa_selecionada == "Todas":
        df_faturamento_pessoal = df_faturamento[(df_faturamento['data'].dt.year == ano_selecionado)]
        df_despesa_pessoal = df_despesa[(df_despesa['data'].dt.year == ano_selecionado) & (df_despesa['tipo'].str.contains("Folha de Pagamento"))]

        df_faturamento_pessoal_filtrado = df_faturamento_pessoal.groupby(df_faturamento_pessoal['data'].dt.month)['valor'].sum(numeric_only=True).reset_index()
        df_despesa_pessoal_filtrado = df_despesa_pessoal.groupby(df_despesa_pessoal['data'].dt.month)['valor'].sum(numeric_only=True).reset_index()

        merged_data_filtered_pessoal = pd.merge(df_faturamento_pessoal_filtrado, df_despesa_pessoal_filtrado, on='data', how='left', suffixes=('_faturamento', '_despesa'))
        merged_data_filtered_pessoal = merged_data_filtered_pessoal.fillna(0)
    else:
        df_faturamento_pessoal = df_faturamento[(df_faturamento['data'].dt.year == ano_selecionado) & df_faturamento['empresa'].str.contains(f"{empresa_selecionada}")]
        df_despesa_pessoal = df_despesa[(df_despesa['data'].dt.year == ano_selecionado) & (df_despesa['empresa'].str.contains(f"{empresa_selecionada}")) & (df_despesa['tipo'].str.contains("Folha de Pagamento"))]

        df_faturamento_pessoal_filtrado = df_faturamento_pessoal.groupby(df_faturamento_pessoal['data'].dt.month)['valor'].sum(numeric_only=True).reset_index()
        df_despesa_pessoal_filtrado = df_despesa_pessoal.groupby(df_despesa_pessoal['data'].dt.month)['valor'].sum(numeric_only=True).reset_index()

        merged_data_filtered_pessoal = pd.merge(df_faturamento_pessoal_filtrado, df_despesa_pessoal_filtrado, on='data', how='left', suffixes=('_faturamento', '_despesa'))
        merged_data_filtered_pessoal = merged_data_filtered_pessoal.fillna(0)

    merged_data_filtered_pessoal['percentage_pessoal'] = (merged_data_filtered_pessoal['valor_despesa'] / merged_data_filtered_pessoal['valor_faturamento']) * 100
    merged_data_filtered_pessoal['percentage_pessoal'] = merged_data_filtered_pessoal['percentage_pessoal'].fillna(0)
    merged_data_filtered_pessoal['percentage_pessoal_str'] = merged_data_filtered_pessoal['percentage_pessoal'].apply(lambda x: f"{x:.2f}%")
    merged_data_filtered_pessoal['mês'] = merged_data_filtered_pessoal['data'].apply(lambda x: f"{util.mapa_meses[x]}/{ano_selecionado}")

    grafico_bar_faturamento = go.Bar(
        x=merged_data_filtered_pessoal['mês'], 
        y=merged_data_filtered_pessoal['valor_faturamento'], 
        name="Receita Total",
        marker_color=util.barra_cinza_claro,
        yaxis='y1'
    )

    grafico_pessoal = go.Scatter(
        x=merged_data_filtered_pessoal['mês'],
        y=merged_data_filtered_pessoal['percentage_pessoal'],
        mode='lines+markers+text',
        name='Despesa com Pessoal / Receita Total',
        text=merged_data_filtered_pessoal['percentage_pessoal_str'],
        textposition='top center',
        marker=dict(color=util.barra_vermelha),
        yaxis='y2'  
    )

    fig_pessoal = go.Figure(data=[grafico_bar_faturamento, grafico_pessoal])

    main_title = f'-DESPESA COM PESSOAL / RECEITA TOTAL ({ano_selecionado})'
    title = f"{main_title} | {subtitle}"

    fig_pessoal.update_layout(
        title=title,
        title_font_color="rgb(98,83,119)",
        title_font_size=15,
        xaxis=dict(
            title="",
            mirror=True,
            showline=True,
            linecolor="Grey",
            linewidth=0.5,
            gridcolor='lightgrey',
            ticks='outside'
        ),
        yaxis=dict(
            title='Receita Total',
            mirror=True,
            showline=True,
            linecolor="Grey",
            linewidth=0.5,
            gridcolor='lightgrey',
            ticks='outside',
            range=[0, merged_data_filtered_pessoal['valor_faturamento'].max() * 1.2],  
            dtick=10000
        ),
        yaxis2=dict(
            title='Despesa com Pessoal (%)',
            overlaying='y',
            side='right',
            range=[0, 100],  
            tickvals=[0, 20, 40, 60, 80, 100], 
            tickformat='.0f',  
            showline=True,
            linecolor="Grey",
            linewidth=0.5,
            ticks='outside',
            gridcolor='lightgrey',
            showgrid=False
        ),
        legend=dict(x=0.1, y=1.1, orientation='h'),
        margin=dict(l=0, r=0, t=70, b=0),
        height=320,
        font=dict(size=14, color='rgb(98,83,119)'),
    )

    c101.plotly_chart(fig_pessoal, use_container_width=True, automargin=True)


    ########################################################################################
    ####### GRÁFICO DESPESA ADMINISTRATIVA #################################################
    ########################################################################################
    if empresa_selecionada == "Todas":
        df_faturamento_administrativa = df_faturamento[(df_faturamento['data'].dt.year == ano_selecionado)]
        df_despesa_administrativa = df_despesa[(df_despesa['data'].dt.year == ano_selecionado) & (df_despesa['tipo'].str.contains("Administrativa"))]

        df_faturamento_administrativa_filtrado = df_faturamento_administrativa.groupby(df_faturamento_administrativa['data'].dt.month)['valor'].sum(numeric_only=True).reset_index()
        df_despesa_administrativa_filtrado = df_despesa_administrativa.groupby(df_despesa_administrativa['data'].dt.month)['valor'].sum(numeric_only=True).reset_index()

        merged_data_filtered_administrativa = pd.merge(df_faturamento_administrativa_filtrado, df_despesa_administrativa_filtrado, on='data', how='left', suffixes=('_faturamento', '_despesa'))
        merged_data_filtered_administrativa = merged_data_filtered_administrativa.fillna(0)
    else:
        df_faturamento_administrativa = df_faturamento[(df_faturamento['data'].dt.year == ano_selecionado) & df_faturamento['empresa'].str.contains(f"{empresa_selecionada}")]
        df_despesa_administrativa = df_despesa[(df_despesa['data'].dt.year == ano_selecionado) & (df_despesa['empresa'].str.contains(f"{empresa_selecionada}")) & (df_despesa['tipo'].str.contains("Administrativa"))]

        df_faturamento_administrativa_filtrado = df_faturamento_administrativa.groupby(df_faturamento_administrativa['data'].dt.month)['valor'].sum(numeric_only=True).reset_index()
        df_despesa_administrativa_filtrado = df_despesa_administrativa.groupby(df_despesa_administrativa['data'].dt.month)['valor'].sum(numeric_only=True).reset_index()

        merged_data_filtered_administrativa = pd.merge(df_faturamento_administrativa_filtrado, df_despesa_administrativa_filtrado, on='data', how='left', suffixes=('_faturamento', '_despesa'))
        merged_data_filtered_administrativa = merged_data_filtered_administrativa.fillna(0)

    merged_data_filtered_administrativa['percentage_administrativa'] = (merged_data_filtered_administrativa['valor_despesa'] / merged_data_filtered_administrativa['valor_faturamento']) * 100
    merged_data_filtered_administrativa['percentage_administrativa'] = merged_data_filtered_administrativa['percentage_administrativa'].fillna(0)
    merged_data_filtered_administrativa['percentage_administrativa_str'] = merged_data_filtered_administrativa['percentage_administrativa'].apply(lambda x: f"{x:.2f}%")
    merged_data_filtered_administrativa['mês'] = merged_data_filtered_administrativa['data'].apply(lambda x: f"{util.mapa_meses[x]}/{ano_selecionado}")

    grafico_bar_faturamento = go.Bar(
        x=merged_data_filtered_administrativa['mês'], 
        y=merged_data_filtered_administrativa['valor_faturamento'], 
        name="Receita Total",
        marker_color=util.barra_cinza_claro,
        yaxis='y1'
    )

    grafico_administrativa = go.Scatter(
        x=merged_data_filtered_administrativa['mês'],
        y=merged_data_filtered_administrativa['percentage_administrativa'],
        mode='lines+markers+text',
        name='Despesa Administrativa / Receita Total',
        text=merged_data_filtered_administrativa['percentage_administrativa_str'],
        textposition='top center',
        marker=dict(color=util.barra_vermelha),
        yaxis='y2'  
    )

    fig_administrativa = go.Figure(data=[grafico_bar_faturamento, grafico_administrativa])

    main_title = f'-DESPESA ADMINISTRATIVA / RECEITA TOTAL ({ano_selecionado})'
    title = f"{main_title} | {subtitle}"

    fig_administrativa.update_layout(
        title=title,
        title_font_color="rgb(98,83,119)",
        title_font_size=15,
        xaxis=dict(
            title="",
            mirror=True,
            showline=True,
            linecolor="Grey",
            linewidth=0.5,
            gridcolor='lightgrey',
            ticks='outside'
        ),
        yaxis=dict(
            title='Receita Total',
            mirror=True,
            showline=True,
            linecolor="Grey",
            linewidth=0.5,
            gridcolor='lightgrey',
            ticks='outside',
            range=[0, merged_data_filtered_administrativa['valor_faturamento'].max() * 1.2],  
            dtick=10000
        ),
        yaxis2=dict(
            title='Despesa Administrativa (%)',
            overlaying='y',
            side='right',
            range=[0, 100],  
            tickvals=[0, 20, 40, 60, 80, 100],  
            tickformat='.0f', 
            showline=True,
            linecolor="Grey",
            linewidth=0.5,
            ticks='outside',
            gridcolor='lightgrey',
            showgrid=False
        ),
        legend=dict(x=0.1, y=1.1, orientation='h'),
        margin=dict(l=0, r=0, t=70, b=0),
        height=320,
        font=dict(size=14, color='rgb(98,83,119)'),
    )

    c102.plotly_chart(fig_administrativa, use_container_width=True, automargin=True)


    ########################################################################################
    ####### GRÁFICO DESPESA OPERACIONAL ####################################################
    ########################################################################################
    if empresa_selecionada == "Todas":
        df_faturamento_operacional = df_faturamento[(df_faturamento['data'].dt.year == ano_selecionado)]
        df_despesa_operacional = df_despesa[(df_despesa['data'].dt.year == ano_selecionado) & (df_despesa['tipo'].str.contains("Operacional"))]

        df_faturamento_operacional_filtrado = df_faturamento_operacional.groupby(df_faturamento_operacional['data'].dt.month)['valor'].sum(numeric_only=True).reset_index()
        df_despesa_operacional_filtrado = df_despesa_operacional.groupby(df_despesa_operacional['data'].dt.month)['valor'].sum(numeric_only=True).reset_index()
        
        merged_data_filtered_operacional = pd.merge(df_faturamento_operacional_filtrado, df_despesa_operacional_filtrado, on='data', how='left', suffixes=('_faturamento', '_despesa'))
        merged_data_filtered_operacional = merged_data_filtered_operacional.fillna(0)
    else: 
        df_faturamento_operacional = df_faturamento[(df_faturamento['data'].dt.year == ano_selecionado) & df_faturamento['empresa'].str.contains(f"{empresa_selecionada}")]
        df_despesa_operacional = df_despesa[(df_despesa['data'].dt.year == ano_selecionado) & (df_despesa['empresa'].str.contains(f"{empresa_selecionada}")) & (df_despesa['tipo'].str.contains("Operacional"))]

        df_faturamento_operacional_filtrado = df_faturamento_operacional.groupby(df_faturamento_operacional['data'].dt.month)['valor'].sum(numeric_only=True).reset_index()
        df_despesa_operacional_filtrado = df_despesa_operacional.groupby(df_despesa_operacional['data'].dt.month)['valor'].sum(numeric_only=True).reset_index()

        merged_data_filtered_operacional = pd.merge(df_faturamento_operacional_filtrado, df_despesa_operacional_filtrado, on='data', how='left', suffixes=('_faturamento', '_despesa'))
        merged_data_filtered_operacional = merged_data_filtered_operacional.fillna(0)

    merged_data_filtered_operacional['percentage_operacional'] = (merged_data_filtered_operacional['valor_despesa'] / merged_data_filtered_operacional['valor_faturamento']) * 100
    merged_data_filtered_operacional['percentage_operacional'] = merged_data_filtered_operacional['percentage_operacional'].fillna(0)
    merged_data_filtered_operacional['percentage_operacional_str'] = merged_data_filtered_operacional['percentage_operacional'].apply(lambda x: f"{x:.2f}%")
    merged_data_filtered_operacional['mês'] = merged_data_filtered_operacional['data'].apply(lambda x: f"{util.mapa_meses[x]}/{ano_selecionado}")

    grafico_bar_faturamento = go.Bar(
        x=merged_data_filtered_operacional['mês'], 
        y=merged_data_filtered_operacional['valor_faturamento'], 
        name="Receita Total",
        marker_color=util.barra_cinza_escuro,
        yaxis='y1'
    )

    grafico_operacional = go.Scatter(
        x=merged_data_filtered_operacional['mês'],
        y=merged_data_filtered_operacional['percentage_operacional'],
        mode='lines+markers+text',
        name='Despesa Operacional / Receita Total',
        text=merged_data_filtered_operacional['percentage_operacional_str'],
        textposition='top center',
        marker=dict(color=util.barra_vermelha),
        yaxis='y2' 
    )

    fig_operacional = go.Figure(data=[grafico_bar_faturamento, grafico_operacional])

    main_title = f'-DESPESA OPERACIONAL / RECEITA TOTAL ({ano_selecionado})'
    title = f"{main_title} | {subtitle}"

    fig_operacional.update_layout(
        title=title,
        title_font_color="rgb(98,83,119)",
        title_font_size=15,        
        xaxis=dict(
            title="",
            mirror=True,
            showline=True,
            linecolor="Grey",
            linewidth=0.5,
            gridcolor='lightgrey',
            ticks='outside'
        ),
        yaxis=dict(
            title='Receita Total',
            mirror=True,
            showline=True,
            linecolor="Grey",
            linewidth=0.5,
            gridcolor='lightgrey',
            ticks='outside',
            range=[0, merged_data_filtered_operacional['valor_faturamento'].max() * 1.2],  
        ),
        yaxis2=dict(
            title='Despesa Operacional (%)',
            overlaying='y',
            side='right',
            range=[0, 100],  
            tickvals=[0, 20, 40, 60, 80, 100],  
            tickformat='.0f',  
            showline=True,
            linecolor="Grey",
            linewidth=0.5,
            ticks='outside',
            gridcolor='lightgrey',
            showgrid=False
        ),
        legend=dict(x=0.1, y=1.1, orientation='h'),
        margin=dict(l=0, r=0, t=70, b=0),
        height=320,
        font=dict(size=14, color='rgb(98,83,119)'),
    )

    c103.plotly_chart(fig_operacional, use_container_width=True, automargin=True)

    ########################################################################################
    ####### GRÁFICO LUCRO PREJUIZO #########################################################
    ########################################################################################
    if empresa_selecionada == "Todas":
        df_faturamento_lucro = df_faturamento[(df_faturamento['data'].dt.year == ano_selecionado)]
        df_despesa_lucro = df_despesa[(df_despesa['data'].dt.year == ano_selecionado)]
        
        df_faturamento_lucro_filtrado = df_faturamento_lucro.groupby(df_faturamento_lucro['data'].dt.month)['valor'].sum(numeric_only=True).reset_index()
        df_despesa_lucro_filtrado = df_despesa_lucro.groupby(df_despesa_lucro['data'].dt.month)['valor'].sum(numeric_only=True).reset_index()

        merged_data_filtered_lucro = pd.merge(df_faturamento_lucro_filtrado, df_despesa_lucro_filtrado, on='data', how='left', suffixes=('_faturamento', '_despesa'))
        merged_data_filtered_lucro = merged_data_filtered_lucro.fillna(0)      
    else:
        df_faturamento_lucro = df_faturamento[(df_faturamento['data'].dt.year == ano_selecionado) & df_faturamento['empresa'].str.contains(f"{empresa_selecionada}")]
        df_despesa_lucro = df_despesa[(df_despesa['data'].dt.year == ano_selecionado) & (df_despesa['empresa'].str.contains(f"{empresa_selecionada}"))]
        
        df_faturamento_lucro_filtrado = df_faturamento_lucro.groupby(df_faturamento_lucro['data'].dt.month)['valor'].sum(numeric_only=True).reset_index()
        df_despesa_lucro_filtrado = df_despesa_lucro.groupby(df_despesa_lucro['data'].dt.month)['valor'].sum(numeric_only=True).reset_index()

        merged_data_filtered_lucro = pd.merge(df_faturamento_lucro_filtrado, df_despesa_lucro_filtrado, on='data', how='left', suffixes=('_faturamento', '_despesa'))
        merged_data_filtered_lucro = merged_data_filtered_lucro.fillna(0)

    merged_data_filtered_lucro['mês'] = merged_data_filtered_lucro['data'].apply(lambda x: f"{util.mapa_meses[x]}/{ano_selecionado}")
    merged_data_filtered_lucro['lucro'] = merged_data_filtered_lucro['valor_faturamento'] -  merged_data_filtered_lucro['valor_despesa']
    merged_data_filtered_lucro['lucro_str'] = merged_data_filtered_lucro['lucro'].apply(lambda x: f"R$ {x:,.2f}".replace('.', '|').replace(',', '.').replace('|', ','))
    merged_data_filtered_lucro['lucro_percent_str'] = ((merged_data_filtered_lucro['lucro'] / merged_data_filtered_lucro['valor_faturamento']) * 100).apply(lambda x: f"{x:.2f}%")
    merged_data_filtered_lucro['color'] = np.where(merged_data_filtered_lucro["lucro"]<0, util.barra_vermelha, util.barra_azul)

    main_title = f'-LUCRO / PREJUÍZO ({ano_selecionado})'
    title = f"{main_title} | {subtitle}"

    grafico_lucro = px.bar(
        merged_data_filtered_lucro, 
        x='mês', 
        y='lucro',
        text = merged_data_filtered_lucro['lucro_str'],    
        color_discrete_sequence = [merged_data_filtered_lucro['color']],
        title=title,
        labels={'mês': '', 'lucro': 'Lucro'}
    )

    grafico_lucro.update_yaxes(showline=True, linecolor="Grey", linewidth=0.1, gridcolor='lightgrey',dtick=2000)
    grafico_lucro.update_xaxes(showline=True, linecolor="Grey", linewidth=0.1, gridcolor='lightgrey')
    grafico_lucro.update_layout(margin=dict(l=0, r=0, t=50, b=0), height=400, barmode='stack', font=dict(size=13, color='white'), title_font_color="rgb(98,83,119)", title_font_size=15,)
    

    c104.plotly_chart(grafico_lucro, use_container_width=True, automargin=True)


    ########################################################################################
    ####### GRÁFICO PANORAMA ANUAL #########################################################
    ########################################################################################
    df_faturamento['ano'] = df_faturamento['data'].dt.year
    df_despesa['ano'] = df_despesa['data'].dt.year

    if empresa_selecionada != "Todas":
        faturamento_grouped = df_faturamento.groupby(['ano', 'empresa'])['valor'].sum().reset_index()
        despesas_grouped = df_despesa.groupby(['ano', 'empresa'])['valor'].sum().reset_index()

        faturamento_grouped = faturamento_grouped[faturamento_grouped['empresa'] == empresa_selecionada]
        despesas_grouped = despesas_grouped[despesas_grouped['empresa'] == empresa_selecionada]

        faturamento_long = faturamento_grouped.melt(id_vars=['ano', 'empresa'], value_vars='valor', value_name='valor_melted', var_name='Categoria')
        faturamento_long['Categoria'] = 'Faturamento'

        despesas_long = despesas_grouped.melt(id_vars=['ano', 'empresa'], value_vars='valor', value_name='valor_melted', var_name='Categoria')
        despesas_long['Categoria'] = 'Despesas'

        data = pd.concat([faturamento_long, despesas_long], ignore_index=True)

        diferencas = faturamento_grouped.set_index(['ano', 'empresa'])['valor'] - despesas_grouped.set_index(['ano', 'empresa'])['valor']
        diferencas = diferencas.reset_index()
        diferencas.columns = ['ano', 'empresa', 'Diferenca']
    else:
        faturamento_grouped = df_faturamento.groupby(['ano'])['valor'].sum().reset_index()
        despesas_grouped = df_despesa.groupby(['ano'])['valor'].sum().reset_index()

        faturamento_long = faturamento_grouped.melt(id_vars=['ano'], value_vars='valor', value_name='valor_melted', var_name='Categoria')
        faturamento_long['Categoria'] = 'Faturamento'

        despesas_long = despesas_grouped.melt(id_vars=['ano'], value_vars='valor', value_name='valor_melted', var_name='Categoria')
        despesas_long['Categoria'] = 'Despesas'

        data = pd.concat([faturamento_long, despesas_long], ignore_index=True)

        diferencas = faturamento_grouped.set_index(['ano'])['valor'] - despesas_grouped.set_index(['ano'])['valor']
        diferencas = diferencas.reset_index()
        diferencas.columns = ['ano', 'Diferenca']        

    color_map = {
        "Faturamento": util.barra_azul,
        "Despesas": util.barra_vermelha
    }

    main_title = f'-BALANCETE ANUAL'
    title = f"{main_title} | {subtitle}"

    grafico_ano = px.bar(data, x='ano', y='valor_melted', color='Categoria', barmode='group', title=title, 
                        labels={'valor_melted': 'Faturamento | Despesa'}, color_discrete_map=color_map)

    for trace in grafico_ano.data:
        trace.text = [util.formata_para_brl(val) for val in trace.y]
        trace.textposition = 'inside'

    for _, row in diferencas.iterrows():
        faturamento_valor = faturamento_grouped[faturamento_grouped['ano'] == row['ano']]['valor'].iloc[0]
        diferenca_valor = row['Diferenca']
        percent_diferenca = (diferenca_valor / faturamento_valor) * 100

        grafico_ano.add_annotation(
            x=row['ano'],
            y=faturamento_valor,
            text=f"<b>DRE: {util.formata_para_brl(diferenca_valor)}</b><br> ({percent_diferenca:.2f}%)",
            showarrow=True,
            arrowhead=1,
            ax=0,
            ay=-40,
            font=dict(size=12)
        )

    grafico_ano.update_layout(
        margin=dict(l=0, r=0, t=50, b=0),
        height=400,
        xaxis_title="",
        font=dict(size=13, color='rgb(98,83,119)'),
        title_font_color="rgb(98,83,119)",
        title_font_size=15,
    )

    grafico_ano.update_yaxes(
        showline=True,
        linecolor="Grey",
        linewidth=0.5,
        gridcolor='lightgrey',
        dtick=50000
    )

    grafico_ano.update_xaxes(
        showline=True,
        linecolor="Grey",
        linewidth=0.5,
        gridcolor='lightgrey'
    )

    c105.plotly_chart(grafico_ano, use_container_width=True, automargin=True)

    ##########################################################################################
    ####### GRAFICO RESUMO RELAÇÃO FATURAMENTO DESPESA #######################################
    ##########################################################################################
    if empresa_selecionada == "Todas":
        df_despesa_filtrado = df_despesa[df_despesa['data'].dt.year == ano_selecionado]
    else:
        df_despesa_filtrado = df_despesa[(df_despesa['data'].dt.year == ano_selecionado) & (df_despesa['empresa'].str.contains(f"{empresa_selecionada}"))]

    despesa_por_mes = df_despesa_filtrado.groupby(df_despesa_filtrado['data'].dt.month)['valor'].sum(numeric_only=True).reset_index()
    despesa_por_mes['mes_str'] = despesa_por_mes['data'].apply(lambda x: util.mapa_meses[x])

    despesa_por_mes.rename(columns={"valor": "Despesa"}, inplace=True)


    grafico_despesa = go.Figure()
    grafico_despesa.add_trace(go.Bar(
        x=despesa_por_mes['mes_str'],
        y=despesa_por_mes['Despesa'],
        text=despesa_por_mes['Despesa'].apply(lambda x: f"R$ {x:,.2f}".replace('.', '|').replace(',', '.').replace('|', ',')),
        name='Despesa',
        marker_color=util.barra_vermelha,
        insidetextfont=dict(size=14, color="white"),
        outsidetextfont=dict(size=14, color=util.barra_azul_escuro)
    ))

    grafico_despesa.update_layout(
        barmode='overlay',
        height=400,
        xaxis_title="",
        font=dict(size=13, color='rgb(249,250,255)')
    )

    grafico_despesa.update_yaxes(showline=True,linecolor="Grey",linewidth=0.5,gridcolor='lightgrey',range=[0, grouped_mensal['Faturado'].max() * 1.3], dtick=5000)
    grafico_despesa.update_xaxes(showline=True,linecolor="Grey",linewidth=0.5,gridcolor='lightgrey')

    if empresa_selecionada == "Todas":
        df_faturamento_filtrado = df_faturamento[df_faturamento['data'].dt.year == ano_selecionado]
    else:
        df_faturamento_filtrado = df_faturamento[(df_faturamento['data'].dt.year == ano_selecionado) & (df_faturamento['empresa'].str.contains(f"{empresa_selecionada}"))]

    faturamento_por_mes = df_faturamento_filtrado.groupby(df_faturamento_filtrado['data'].dt.month)['valor'].sum(numeric_only=True).reset_index()
    faturamento_por_mes['mes_str'] = faturamento_por_mes['data'].apply(lambda x: util.mapa_meses[x])

    data_final_disponivel = df_faturamento_filtrado['data'].max()
    ultimo_mes = data_final_disponivel.month
    dias_no_mes = data_final_disponivel.days_in_month
    dias_ate_data_final = data_final_disponivel.day

    dados_ultimo_mes = df_faturamento_filtrado[df_faturamento_filtrado['data'].dt.month == ultimo_mes]
    total_ultimo_mes = dados_ultimo_mes['valor'].sum(numeric_only=True)

    valor_previsao = (total_ultimo_mes / dias_ate_data_final) * dias_no_mes

    faturamento_por_mes.loc[faturamento_por_mes['data'] == ultimo_mes, 'previsao'] = valor_previsao
    faturamento_por_mes.fillna({'previsao': 0}, inplace=True)

    faturamento_por_mes.rename(columns={"valor": "Faturado", "previsao": "Previsto"}, inplace=True)

    grafico_faturamento = px.bar(faturamento_por_mes, 
        x=faturamento_por_mes['mes_str'], 
        y=['Previsto', 'Faturado'],
        color_discrete_sequence=["#c6ccd2", util.barra_azul])  

    
    grafico_faturamento.data[0].text = faturamento_por_mes['Previsto'].apply(util.formata_para_brl).tolist()
    grafico_faturamento.data[1].text = faturamento_por_mes['Faturado'].apply(util.formata_para_brl).tolist()


    merged_data = faturamento_por_mes.merge(despesa_por_mes, on='data', how='left', suffixes=('_faturamento', '_despesa'))
    merged_data['diferenca'] = merged_data['Faturado'] - merged_data['Despesa']

    for trace in grafico_faturamento['data']:
        trace['x'] = [util.meses_mapa[mes.split('/')[0]] - 0.15 for mes in trace['x']]
        trace['width'] = 0.3
        grafico_despesa.add_trace(trace)

    for trace in grafico_despesa['data']:
        if trace['name'] == 'Despesa':
            trace['x'] = [util.meses_mapa[mes] + 0.15 for mes in trace['x']]
            trace['width'] = 0.3

    tickvals = list(util.meses_mapa.values())
    diferencas = [util.formata_para_brl(merged_data.loc[merged_data['data'] == val, 'diferenca'].values[0]) if val in merged_data['data'].values else 'N/A' for val in tickvals]
    ticktext = [f"{mes}\n{dif}" for mes, dif in zip(list(util.meses_mapa.keys()), diferencas)]

    main_title = f'-RELAÇÃO FATURAMENTO | DESPESAS MENSAL ({ano_selecionado})'
    title = f"{main_title} | {subtitle}"

    nomes_meses = list(util.meses_mapa.keys())

    nomes_meses_com_ano = [f"{mes}/{ano_selecionado}" for mes in nomes_meses]

    grafico_despesa.update_layout(
        xaxis=dict(tickvals=tickvals, ticktext=nomes_meses_com_ano),
        legend_title_text='Faturamento | Despesas',
        title=title,
        font=dict(size=13, color='rgb(249,250,255)'),
        yaxis_title="Faturamento | Despesas",
        margin=dict(l=0, r=0, t=50, b=0),
        title_font_color="rgb(98,83,119)"
    )

    for mes, val in util.meses_mapa.items():
        diferenca_values = merged_data_filtered_lucro.loc[merged_data_filtered_lucro['data'] == val, 'lucro'].values
        faturamento_values = merged_data.loc[merged_data['data'] == val, 'Faturado'].values

        if diferenca_values.size > 0 and faturamento_values.size > 0 and faturamento_values[0] != 0:
            percentage = (diferenca_values[0] / faturamento_values[0]) * 100

            grafico_despesa.add_annotation(
                x=val,
                y=0.965,  
                text=f"<b>DRE: {util.formata_para_brl(diferenca_values[0])}</b>",
                showarrow=False,
                yref="paper",
                xref="x", 
                font=dict(size=12, color="rgb(98,83,119)"),
                align="center"
            )

            grafico_despesa.add_annotation(
                x=val,
                y=0.918,  
                text=f"({percentage:.2f}%)",
                showarrow=False,
                yref="paper",
                xref="x", 
                font=dict(size=12, color="rgb(98,83,119)"),
                align="center"
            )

    for mes, val in util.meses_mapa.items():
        percentage_cmv_val = merged_data_filtered_cmv.loc[merged_data_filtered_cmv['data'] == val, 'percentage_cmv'].values
            
        if percentage_cmv_val.size > 0 and not np.isclose(percentage_cmv_val[0], 0):
            grafico_despesa.add_annotation(
                x=val,
                y=-0.145,  
                text=f"<b>CMV: ({percentage_cmv_val[0]:.2f}%)</b>",
                showarrow=False,
                yref="paper",
                xref="x", 
                font=dict(size=12, color="rgb(98,83,119)"),
                align="center"
            )

    c2.plotly_chart(grafico_despesa, use_container_width=True, automargin=True)

########################################################################################
####### ABA NFEs #########################################################
########################################################################################
if empresa_nome == "MH Refeições":
    with tab_objects[4]:
        st.session_state["aba_ativa"] = "tab4"

        if not st.session_state["autenticado_tab4"]:
            with st.container(border=True):
                placeholder4 = st.empty()  

                with st.form("form_tab4", clear_on_submit=False):
                    codigo4 = st.text_input("Digite o Código de Acesso:", type="password", key="codigo_tab4")
                    submit4 = st.form_submit_button("Entrar")

                if submit4:
                    if codigo4 == CODE:
                        st.session_state["autenticado_tab4"] = True
                        st.session_state["aba_ativa"] = "tab4"
                        st.rerun()
                    else:
                        placeholder4.error("Código Incorreto ❌")
                else:
                    placeholder4.warning("Digite o Código para acessar NFE's")

        else:
        
            arquivos = listar_arquivos_pdfs()

            if not arquivos:
                st.warning("Nenhum arquivo PDF encontrado na pasta.")
            else:
                nomes = [f["name"] for f in arquivos]

                anos_meses = sorted(
                    {(n[:4], n[4:6]) for n in nomes if len(n) >= 6},
                    key=lambda x: (x[0], x[1])
                )

                ultimo_ano, ultimo_mes = anos_meses[-1]

                with st.container(border=True):
                    col_ano_pdf, col_mes_pdf, col_drop_pdf, col_download_pdf = st.columns([2, 2, 3, 2])

                with col_ano_pdf:
                    anos_disp = sorted({a for a, m in anos_meses})
                    ano_sel = st.selectbox(
                        "Selecione o Ano:",
                        anos_disp,
                        index=anos_disp.index(ultimo_ano),
                        key="ano_pdf"
                    )

                with col_mes_pdf:
                    meses_disp = sorted({m for a, m in anos_meses if a == ano_sel})
                    meses_labels = [util.mapa_meses[int(m)] for m in meses_disp]
                    mes_idx_default = meses_disp.index(ultimo_mes) if ano_sel == ultimo_ano else 0

                    mes_sel_label = st.selectbox(
                        "Selecione o Mês:",
                        meses_labels,
                        index=mes_idx_default,
                        key="mes_pdf"
                    )

                    mes_sel = f"{list(util.mapa_meses.keys())[list(util.mapa_meses.values()).index(mes_sel_label)]:02d}"

                nomes_filtrados = [n for n in nomes if n.startswith(f"{ano_sel}{mes_sel}")]
            
                with col_drop_pdf:
                    escolha = st.selectbox(
                        "Selecione um Fechamento: (aaaamm-seq)",
                        nomes_filtrados,
                        key="drop_pdf"
                    )

                with col_download_pdf:
                    st.markdown("<div style='margin-top: 27.8px;'>", unsafe_allow_html=True)
                    idx = nomes.index(escolha)
                    file_id = arquivos[idx]["id"]
                    fh = baixar_pdf(file_id)
                    st.download_button(
                        label="📥 Download NFE",
                        data=fh.getvalue(),
                        file_name=escolha,
                        mime="application/pdf",
                        key="download_pdf_tab4",
                        use_container_width=True
                    )
                    st.markdown("</div>", unsafe_allow_html=True)

                idx = nomes.index(escolha)
                file_id = arquivos[idx]["id"]
                pdf_url = f"https://drive.google.com/file/d/{file_id}/preview"
                with st.container(border=True):
                    st.components.v1.iframe(src=pdf_url, height=450, scrolling=True)