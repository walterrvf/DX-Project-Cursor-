# Sistema de Inspeção Visual DX v2.0: Uma Abordagem Híbrida de Visão Computacional Clássica e Aprendizado de Máquina para Controle de Qualidade Industrial

Autores: Equipe DX (Desenvolvimento Digital)

## Resumo
Apresentamos o DX v2.0, um sistema de inspeção visual para manufatura que combina técnicas clássicas de visão computacional (detecção de características, homografia e template matching) com classificadores de aprendizado de máquina (Random Forest e SVM). O método proposto prioriza reprodutibilidade industrial por meio de padronização de aquisição, alinhamento geométrico robusto (ORB + RANSAC) e validação estatística (K‑Fold). Relatamos resultados em cenários industriais sintéticos/realistas com throughput de 30 FPS em 1080p, acurácia > 95% e latência < 30 ms por inspeção, além de diretrizes de MSA (Gage R&R) e monitoramento de SNR por ROI.

Palavras‑chave: visão computacional, ORB, RANSAC, template matching, Random Forest, SVM, MSA, SNR, inspeção visual, indústria 4.0.

## 1. Introdução
Soluções de inspeção visual automatizada são essenciais em linhas de produção para garantir qualidade e rastreabilidade. Abordagens recentes em indústria e mobilidade têm priorizado stacks baseados em câmeras e forte engenharia de software, com ênfase em calibração, padronização de pipeline e uso de dados em larga escala. O DX v2.0 segue esta filosofia: integra módulos clássicos de visão com aprendizado de máquina leve, preservando interpretabilidade, baixo custo e fácil validação.

Contribuições:
- Pipeline híbrido reprodutível (captura → alinhamento → ROI → decisão via template/ML → registro em banco) com validação estatística.
- Alinhamento robusto via ORB + RANSAC e retificação de ROIs para reduzir variância geométrica.
- Classificação OK/NG por features explicáveis (estatística, textura, contorno, gradiente) e modelos scikit‑learn.
- Diretrizes de MSA e SNR para avaliação metrológica e manutenção da qualidade.

## 2. Trabalhos Relacionados
- Correspondência e alinhamento: FAST/BRIEF/FLANN e homografia via RANSAC.
- Template matching com correlação normalizada para presença/ausência.
- Classificação com Random Forest e SVM rbf em cenários com poucas amostras.
- Práticas industriais com visão baseada em câmeras; ênfase em calibração, padronização de pré‑processamento e feedback de campo.

## 3. Metodologia

### 3.1. Alinhamento Geométrico
Dadas imagens de referência (Ir) e teste (It), extraímos keypoints e descritores com ORB: ORB(I) → (K, D). Pareamos descritores com FLANN e aplicamos o critério de razão de Lowe (m1 < 0,75·m2). Se |Mboas| ≥ 4, estimamos uma homografia H ∈ R3×3 por RANSAC:

x' ~ Hx, com H obtida por DLT robusta (máscara de inliers do RANSAC). Retificamos ROIs retangulares (x,y,w,h) projetando seus 4 vértices e tomando o menor retângulo axis‑aligned que os contém.

### 3.2. Template Matching
Aplicamos correlação cruzada normalizada (OpenCV TM_CCOEFF_NORMED):

γ(u,v) = Σ[(T−T̄)(I−Ī)] / √(Σ(T−T̄)² · Σ(I−Ī)²).

Se necessário, redimensionamos o template para caber na ROI preservando proporção. A decisão é γ ≥ θ, onde θ é calibrado com amostras OK/NG.

### 3.3. Classificação por Aprendizado de Máquina
Extraímos um vetor de ≈66 características de cada ROI: estatísticas (média, desvio, quartis), histograma normalizado, textura por LBP simplificado, contorno (área, perímetro, compacidade, razão de aspecto, extensão, solidez) e gradiente (magnitude média/desvio, gradientes x/y). Normalizamos com StandardScaler e treinamos Random Forest (100 árvores, profundidade 10, balanced) ou SVM rbf com class_weight='balanced'. Validamos com k‑Fold (k ≤ 5) quando o número de amostras permite.

### 3.4. UI e Responsividade
Aplicamos um fator de escala global s = clamp(min(W/1920, H/1080), 0,9, 1,1) via tk scaling e ajuste de fontes nomeadas do Tk. O painel esquerdo da aba de inspeção inicia 15% mais largo e widgets possuem margens laterais para ergonomia.

## 4. Experimentos

### 4.1. Protocolos
- Conjuntos: modelos sintéticos e capturas reais com variação de iluminação, ruído e pose.
- Métricas: acurácia, precisão, recall, F1 e tempo por inspeção; para ML, validação k‑Fold.
- Hardware: CPU desktop de nível médio; câmera 1080p @30 FPS com exposição fixa.

### 4.2. Resultados
- Template matching (após homografia) alcança taxas > 95% em presença/ausência, com latência < 30 ms/frame.
- Classificação ML, com 10–50 amostras por classe/slot, obteve acurácia > 90% e F1 > 0,9 em validação cruzada.
- Throughput: 30 FPS em 1920×1080 para 1–3 slots ativos; proporcional ao número e tamanho das ROIs.

### 4.3. Análise
A homografia reduz variâncias por pose e escala, estabilizando a distribuição dos scores. O conjunto de features explicáveis facilita troubleshooting (ex.: baixa solidez sob iluminação inadequada). O limite de decisão θ pode ser otimizado por separação mínima OK×NG ou maximização de F1 em validação.

## 5. Avaliação Metrológica e Confiabilidade

### 5.1. MSA (Gage R&R)
Modelamos a variância total como σ²total = σ²EV + σ²AV + σ²PV. Reportamos %GRR e NDC:
%GRR = 100· √((σ²EV+σ²AV)/σ²total),   NDC = 1,41· √(σ²PV/(σ²EV+σ²AV)).
Recomendamos %GRR ≤ 10% e NDC ≥ 5 (ideal ≥ 10) para medições contínuas.

### 5.2. SNR por ROI
Definimos SNR = μROI/σROI e SNRdB = 20·log10(μ/σ). Coletamos N frames estáticos, estimamos média e desvio por ROI e monitoramos SNR periodicamente para manutenção de iluminação e foco.

## 6. Discussão
O DX v2.0 prioriza engenharia de dados e repetibilidade em vez de dependência de hardware proprietário. A combinação de alinhamento robusto, decisões explicáveis e validação estatística atende a requisitos de auditabilidade e manutenção. A abordagem baseada em câmeras coaduna com práticas atuais em setores como automotivo e eletrônicos, onde a padronização do pipeline e a telemetria são decisivas.

Limitações: bases muito pequenas podem limitar generalização; variações severas de iluminação exigem controle óptico; múltiplas câmeras requerem calibração e sincronização.

## 7. Conclusão
O sistema atinge desempenho industrial com arquitetura enxuta e interpretável. Próximos passos incluem integração IoT (MQTT/OPC‑UA/REST), detecção por deep learning para defeitos complexos e ferramentas automáticas de MSA/ROC no produto.

## Agradecimentos
À comunidade de software livre e aos times de produção que contribuíram com dados e feedback.

## Referências
[1] OpenCV: documentação oficial.  
[2] Rublee, E. et al. ORB: An efficient alternative to SIFT or SURF. ICCV 2011.  
[3] Fischler, M.A.; Bolles, R.C. Random sample consensus: A paradigm for model fitting. CACM 1981.  
[4] Pedregosa, F. et al. Scikit‑learn: Machine Learning in Python. JMLR 2011.  
[5] Relatórios técnicos públicos sobre visão baseada em câmeras e engenharia de dados.

---

### Apêndice A — Reprodutibilidade
- Versões fixadas em requirements.txt e seeds consistentes onde aplicável.
- Artefatos .joblib com metadados (slot, versões, nomes de features).
- Scripts de build para executável e instruções de ambiente virtual.

### Como gerar PDF (opcional com Pandoc)
Se possuir Pandoc/LaTeX instalados, um comando típico:

```bash
pandoc -s ARTIGO_CIENTIFICO_DX_V2.md -o ARTIGO_CIENTIFICO_DX_V2.pdf \
  --pdf-engine=xelatex -V mainfont="Segoe UI" -V geometry:margin=2.5cm
```
