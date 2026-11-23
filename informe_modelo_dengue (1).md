# Informe técnico: modelo de apoyo a la decisión clínica para dengue en Argentina

## 1. Propósito del proyecto

Desarrollar un modelo de aprendizaje automático como **herramienta de apoyo a la decisión clínica** para pacientes febriles con sospecha de dengue en Argentina.

El objetivo no es reemplazar el diagnóstico etiológico ni automatizar decisiones médicas. El objetivo es ayudar al profesional de salud, en consulta, guardia o contexto de brote, a:

- Estimar una probabilidad de dengue condicionada a las características del paciente.
- Priorizar la solicitud de NS1/qRT-PCR.
- Decidir si corresponde solicitar o repetir un hemograma.
- Identificar pacientes que requieren control clínico precoz o vigilancia estrecha.
- Optimizar el uso de recursos de laboratorio durante períodos de alta demanda.

El sistema debe entregarse como una recomendación probabilística y una prioridad de acción, siempre sujeta a evaluación clínica, epidemiológica y a los algoritmos vigentes de laboratorio.

## 2. Vacío que se pretende cubrir

### Vacío identificado

Existe una brecha en la disponibilidad de herramientas de triage basadas en variables clínicas de primer contacto, adaptadas al contexto argentino, que permitan estimar la probabilidad de dengue y priorizar la confirmación de laboratorio en escenarios de alta demanda, atención primaria o guardia, sin requerir obligatoriamente un hemograma inicial.

### Aporte propuesto

El proyecto propone una herramienta de estratificación clínica que use información disponible en la primera consulta para orientar decisiones operativas:

- ¿Qué probabilidad tiene este paciente de tener dengue?
- ¿Corresponde priorizar NS1/qRT-PCR?
- ¿Conviene solicitar un hemograma ahora?
- ¿Debe reevaluarse precozmente o repetirse el hemograma?
- ¿Qué prioridad tiene dentro de una cola de estudios de laboratorio?

No se propone un sistema que “diagnostique dengue sin laboratorio” ni que “descarte dengue”. La confirmación etiológica sigue requiriendo NS1, qRT-PCR, aislamiento viral o serología según el momento de la consulta y el algoritmo vigente.

## 3. Justificación clínica y epidemiológica

En Argentina, la sospecha de dengue se basa en criterios clínicos: fiebre aguda, habitualmente de 2 a 7 días de evolución, sin síntomas de vías aéreas superiores ni otra etiología definida, acompañada de al menos dos manifestaciones compatibles, entre ellas cefalea, mialgia y exantema.

La leucopenia y la plaquetopenia también figuran como criterios complementarios de sospecha cuando están disponibles. Esto respalda que las variables `WBC_Count` y `Platelet_Count` sean clínicamente pertinentes, pero no convierte al hemograma en una prueba diagnóstica definitiva.

La duración de la fiebre es especialmente relevante para la estrategia de laboratorio:

- Entre 0 y 3 días de evolución: se recomiendan métodos directos, como NS1, qRT-PCR o aislamiento viral.
- Entre 4 y 6 días: se recomienda combinar un método indirecto con al menos un método directo.
- Desde 7 días: se priorizan IgM y pruebas de neutralización.

Por lo tanto, el modelo no debe limitarse a predecir “dengue sí/no”. Debe incorporar la duración de la fiebre para recomendar la secuencia diagnóstica más adecuada.

## 4. Variables disponibles

| Variable | Tipo | Rol clínico propuesto |
|---|---|---|
| `Fever` | Binaria / categórica | Condición de entrada del triage; fiebre es necesaria para la sospecha, pero altamente inespecífica |
| `Fever_Duration` | Numérica, en días | Determina la ventana diagnóstica y ayuda a diferenciar fase temprana de fase tardía |
| `Age` | Numérica | Puede modificar la presentación clínica y el riesgo; no es un signo de alarma por sí mismo |
| `Gender` | Categórica | Puede capturar diferencias de presentación o de acceso, pero no debe interpretarse causalmente |
| `Myalgia` | Binaria | Síntoma compatible con dengue y presente en la definición argentina de caso sospechoso |
| `Headache` | Binaria | Síntoma compatible con dengue y presente en la definición argentina de caso sospechoso |
| `Rash` | Binaria | Síntoma compatible, aunque no específico de dengue |
| `Platelet_Count` | Numérica | Aumenta la probabilidad de dengue si está baja; su valor depende del día de evolución y de la presencia de mediciones previas |
| `WBC_Count` | Numérica | La leucopenia temprana aumenta la sospecha; la tendencia es más informativa que un valor aislado |

## 5. Alcance y límites del modelo

### Lo que el modelo puede hacer

Con las variables disponibles, el modelo puede:

- Estimar una probabilidad de dengue condicionada a las características del paciente.
- Estratificar pacientes en prioridad baja, intermedia o alta para confirmación de laboratorio.
- Recomendar la ventana diagnóstica más adecuada según la duración de la fiebre.
- Identificar pacientes que requieren hemograma, repetición de hemograma o control precoz.
- Apoyar la priorización de recursos durante brotes.

### Lo que el modelo no debe hacer

El modelo no debe:

- Confirmar dengue sin NS1, qRT-PCR, aislamiento viral o serología.
- Descartar dengue.
- Reemplazar el examen físico.
- Reemplazar la evaluación de signos de alarma.
- Decidir alta, internación o derivación por sí solo.
- Predecir choque, dengue grave o deterioro con precisión suficiente usando una única medición de laboratorio.
- Diferenciar dengue de chikungunya, Zika, influenza, COVID-19 u otras causas febriles con especificidad suficiente.

### Decisión sobre calibración

En una primera instancia, el proyecto trabajará con la probabilidad estimada por el modelo **sin calibración posterior**. Es decir, el modelo entregará una salida como `P(dengue | características del paciente)`, pero no se asumirá que esa probabilidad esté calibrada.

La calibración se evaluará en una etapa futura de validación, comparando las probabilidades predichas con las frecuencias observadas de dengue confirmado. Si la evaluación muestra mala calibración, se considerará incorporar métodos de calibración posterior, como regresión isotónica o Platt scaling.

Esta decisión delimita claramente el alcance inicial del trabajo:

- El modelo entrega una probabilidad estimada de dengue.
- La calibración no es una salida clínica ni una propiedad asumida del modelo.
- La calibración es un criterio de evaluación futuro.
- La incorporación de calibración posterior será decidida según los resultados de validación.

## 6. Hipótesis general

En pacientes febriles con sospecha de dengue atendidos en Argentina durante un brote, un modelo de aprendizaje automático basado en variables clínicas de primer contacto —fiebre, duración de la fiebre, edad, sexo, mialgia, cefalea y rash— puede estimar una probabilidad de dengue condicionada a las características del paciente y priorizar la confirmación por NS1/qRT-PCR, sin requerir hemograma en la evaluación inicial.

En una primera instancia, el modelo entregará una probabilidad estimada de dengue sin calibración posterior. La calibración se evaluará como una etapa futura de validación y, si corresponde, se incorporará mediante métodos de calibración posterior.

El modelo se concibe como una herramienta de triage y apoyo a la decisión clínica, no como un sistema diagnóstico autónomo.

## 7. Hipótesis específicas

### Hipótesis 1: modelo clínico sin laboratorio

En pacientes febriles con sospecha de dengue en Argentina, un modelo basado exclusivamente en fiebre, duración de la fiebre, edad, sexo, mialgia, cefalea y rash puede estratificar la probabilidad de dengue y priorizar la confirmación por NS1/qRT-PCR.

**Desenlace primario:** dengue confirmado por NS1, qRT-PCR, aislamiento viral o serología, según el algoritmo vigente.

**Resultado esperado:** el modelo clínico-only supera a una regla simple basada exclusivamente en la definición de caso sospechoso para estratificar probabilidad y prioridad de estudio.

### Hipótesis 2: valor añadido del hemograma único

La incorporación de `Platelet_Count` y `WBC_Count`, cuando están disponibles, mejora la discriminación y la utilidad clínica del modelo para estimar la probabilidad de dengue y priorizar la confirmación por NS1/qRT-PCR; su efecto sobre la calibración se evaluará en la etapa futura de validación.

**Comparación central:** modelo clínico-only frente a modelo clínico más hemograma único.

**Resultado esperado:** el modelo con plaquetas y WBC mejora la discriminación, especialmente en pacientes con recuentos alterados o en fases más avanzadas de la enfermedad.

### Hipótesis 3: utilidad diferencial del hemograma

El aporte de plaquetas y WBC es mayor para identificar pacientes que requieren vigilancia estrecha o repetición de hemograma que para diferenciar dengue de otras arbovirosis.

**Justificación:** plaquetas y WBC aportan información sobre fase de enfermedad y riesgo hematológico, pero no eliminan el solapamiento clínico con chikungunya, Zika, influenza, COVID-19 u otras causas de fiebre.

### Hipótesis 4: límite explícito del modelo

Sin mediciones seriadas de plaquetas, WBC, hematocrito y signos de alarma, el modelo no puede predecir de manera confiable el deterioro a dengue grave o choque.

Por ello, la salida del modelo debe limitarse a:

- Probabilidad de dengue.
- Prioridad de NS1/qRT-PCR.
- Necesidad de hemograma o repetición de hemograma.
- Necesidad de control clínico precoz.

## 8. Diseño experimental propuesto

### Modelos comparables

| Modelo | Variables incluidas | Pregunta clínica | Rol en el estudio |
|---|---|---|---|
| M0: regla clínica argentina | Fiebre, duración, síntomas y criterios de caso sospechoso | ¿Cumple criterios de sospecha? | Línea base institucional |
| M1: síntomas only | Fiebre, `Fever_Duration`, edad, sexo, mialgia, cefalea, rash | ¿Qué probabilidad de dengue y prioridad de NS1/qRT-PCR sin laboratorio? | Aporte principal |
| M2: síntomas + hemograma único | M1 + `Platelet_Count` + `WBC_Count` | ¿Mejora la estratificación cuando el hemograma está disponible? | Análisis de valor añadido |
| M3: modelo con serie temporal | M2 + mediciones repetidas de plaquetas y WBC | ¿La dinámica hematológica mejora la predicción de deterioro? | Límite futuro; no viable con el dataset actual |

### Comparación principal

La comparación principal del trabajo será:

**M1: síntomas only vs. M2: síntomas + plaquetas + WBC**

Esta comparación responde a la pregunta central: ¿cuánto aporta el hemograma único cuando ya está disponible, y qué se pierde si se busca una herramienta que no requiera laboratorio en la primera consulta?

### Variables derivadas recomendadas

Para el modelo M2 se recomienda crear variables interpretables:

- `Thrombocytopenia`: plaquetas por debajo de un umbral clínico local.
- `Leukopenia`: WBC por debajo de un umbral clínico local.
- `Platelet_WBC_Ratio`: relación entre plaquetas y leucocitos.
- `Fever_Duration_x_Platelets`: interacción entre día de evolución y plaquetas.
- `Fever_Duration_x_WBC`: interacción entre día de evolución y leucocitos.
- Indicador de contexto de brote, si el dataset permite identificarlo.

La interacción entre día de evolución y recuentos hematológicos es importante porque una plaquetopenia en el día 2 no tiene la misma interpretación clínica que la misma cifra en el día 5.

## 9. Algoritmos propuestos

Se recomienda evaluar al menos los siguientes modelos:

- Regresión logística regularizada, como línea base interpretable.
- Árbol de decisión, como modelo transparente.
- Random Forest.
- XGBoost.
- LightGBM.
- Modelo aditivo interpretable, como GAM o EBM, si se busca máxima explicabilidad.

La elección final no debe basarse únicamente en AUROC. Debe priorizar interpretabilidad, estabilidad y utilidad clínica.

## 10. Desenlaces propuestos

### Desenlace 1: dengue confirmado

Variable binaria:

- Dengue confirmado por NS1, qRT-PCR, aislamiento viral o serología.
- No dengue confirmado.

Este es el desenlace principal para la hipótesis de probabilidad de dengue.

### Desenlace 2: prioridad de laboratorio

Variable ordinal o multinomial:

- Prioridad alta para NS1/qRT-PCR.
- Prioridad intermedia.
- Prioridad baja o estudio diferido.

Este desenlace es más útil que una clasificación binaria porque incorpora la duración de la fiebre y la disponibilidad local de pruebas.

### Desenlace 3: necesidad de vigilancia estrecha

Variable binaria u ordinal:

- Control clínico precoz.
- Repetición de hemograma.
- Derivación a guardia.
- Reevaluación en 24–48 horas.

Este desenlace debe definirse de forma explícita antes del modelado. No debe confundirse con “dengue grave”, “choque” o “internación”.

## 11. Métricas de evaluación

### Métricas de discriminación

- AUROC.
- AUPRC, especialmente útil si la prevalencia de dengue es baja.
- Sensibilidad.
- Especificidad.
- VPP.
- VPN.

### Métricas de calibración — evaluación futura

En la primera etapa no se aplicará calibración posterior. La calibración se evaluará posteriormente mediante:

- Brier score.
- Curva de calibración.
- Calibration-in-the-large.
- Pendiente de calibración.

Si la evaluación muestra mala calibración, se considerará incorporar métodos de calibración posterior, como regresión isotónica o Platt scaling.

### Métricas de utilidad clínica

- Número necesario de estudios para identificar un caso confirmado.
- Reducción estimada de estudios innecesarios.
- Decision curve analysis.
- Análisis por umbral clínico, no solo el umbral que maximiza Youden.
- Análisis estratificado por día de fiebre.

### Análisis de ablation

Se recomienda evaluar:

- Modelo completo.
- Modelo sin `Platelet_Count`.
- Modelo sin `WBC_Count`.
- Modelo sin ambos recuentos.
- Modelo sin `Fever_Duration`.
- Modelo sin síntomas.
- Modelo sin edad y sexo.

Esto permitirá cuantificar el aporte incremental de cada bloque de información.

## 12. Riesgos metodológicos

### Fuga de información

Debe verificarse que ninguna variable del outcome, como resultado de NS1, qRT-PCR, serología, fecha de confirmación o gravedad final, esté incluida como predictor.

### Sesgo de selección

El dataset puede estar sesgado hacia pacientes que ya fueron estudiados, internados o atendidos en centros con mayor capacidad diagnóstica. Esto puede inflar artificialmente el desempeño del modelo.

### Prevalencia variable

El VPP y el VPN dependen fuertemente de la prevalencia local. Un modelo con buena AUROC puede tener VPP bajo si la prevalencia de dengue es baja.

### Solapamiento clínico

Los síntomas de dengue se superponen con chikungunya, Zika, influenza, COVID-19 y otras infecciones. Por ello, el modelo debe presentarse como herramienta de priorización y no como diagnóstico.

### Falta de serie temporal

Sin mediciones repetidas, el modelo no puede capturar la velocidad de caída de plaquetas ni la evolución del WBC. Esto limita su capacidad para predecir deterioro.

### Validación insuficiente

El modelo debe validarse temporalmente, externamente o prospectivamente. Si no es posible, debe presentarse como prueba de concepto y no como herramienta lista para implementación clínica.

## 13. Salida clínica propuesta

La interfaz no debería mostrar únicamente “dengue: positivo/negativo”. Debería entregar una salida como la siguiente:

```text
Probabilidad estimada de dengue: 68%
Prioridad de confirmación: alta
Prueba recomendada: NS1 y/o qRT-PCR
Hemograma: solicitar hoy
Control clínico: reevaluar en 24–48 horas
Alerta: el resultado no reemplaza la evaluación clínica ni los signos de alarma
```

La salida debe incluir siempre:

- Probabilidad estimada de dengue.
- Nivel de prioridad.
- Acción recomendada.
- Momento sugerido de reevaluación.
- Advertencia explícita de que no reemplaza el juicio clínico.

## 14. Conclusión para el equipo

El proyecto es viable y tiene un vacío clínico bien delimitado: desarrollar una herramienta argentina de triage y priorización para dengue basada en variables de primer contacto.

La propuesta más sólida es comparar un modelo clínico-only con un modelo que incorpora hemograma único. Esto permite demostrar tanto la utilidad de una herramienta sin laboratorio como el valor incremental de plaquetas y WBC cuando están disponibles.

La predicción de deterioro no debe ser el objetivo principal del modelo actual, porque requiere información longitudinal y clínica que el dataset no contiene. En cambio, el modelo puede aportar valor real al priorizar confirmación de laboratorio, solicitar hemograma y definir controles precoces.

La hipótesis debe formularse como apoyo probabilístico a la decisión clínica, no como diagnóstico autónomo.

## 15. Fuentes y evidencia de respaldo

### Guías y documentos normativos

1. Ministerio de Salud de la Nación Argentina. *Algoritmo clínico dengue 2024*. https://www.argentina.gob.ar/sites/default/files/2024/09/algoritmo_clinico_dengue_2024.pdf  
2. Ministerio de Salud de la Nación Argentina. *Orientaciones para la vigilancia, atención de casos y estrategia de laboratorio para dengue y chikungunya*. https://www.argentina.gob.ar/sites/default/files/2019/10/orientaciones-vigilancia-en-brotes-dengue_1842024.pdf  
3. Ministerio de Salud de la Nación Argentina. *Algoritmo dengue laboratorio*. https://www.argentina.gob.ar/sites/default/files/2019/10/algoritmos-dengue-laboratorio-11-2024.pdf  
4. Ministerio de Salud de la Nación Argentina. *Síntomas y signos de alarma — Dengue*. https://www.argentina.gob.ar/sintomas-y-signos-de-alarma  
5. Ministerio de Salud de la Provincia de Buenos Aires. *¿Cuándo sospechar dengue o chikungunya?* https://www.ms.gba.gov.ar/sitios/media/files/2024/03/Cu%C3%A1ndo-sospechar-dengue-o-chikungunya.pdf  
6. Organización Mundial de la Salud. *Dengue: guidelines for diagnosis, treatment, prevention and control*. https://iris.who.int/bitstream/handle/10665/76887/9789241504713_eng.pdf  
7. Organización Mundial de la Salud. *Laboratory testing for dengue virus*. https://iris.who.int/server/api/core/bitstreams/7f5d3370-9990-426d-a848-9225f3ecbc48/content  
8. Centers for Disease Control and Prevention. *Clinical Testing Guidance for Dengue*. https://www.cdc.gov/dengue/hcp/diagnosis-testing/index.html  
9. Centers for Disease Control and Prevention. *Guidelines for Classifying Dengue*. https://www.cdc.gov/dengue/hcp/clinical-signs/guidelines.html  

### Artículos sobre modelos de machine learning

10. Ming DK, et al. *The Diagnosis of Dengue in Patients Presenting With Acute Febrile Illness Using Supervised Machine Learning and Impact of Seasonality*. *Frontiers in Digital Health*. 2022;4:849641. https://pmc.ncbi.nlm.nih.gov/articles/PMC8963938/  
11. Madewell ZJ, et al. *Machine learning for predicting severe dengue in Puerto Rico*. *Infectious Diseases of Poverty*. 2025;14:5. https://pmc.ncbi.nlm.nih.gov/articles/PMC11796212/  
12. Yilmaz E, et al. *Tree-Based Machine Learning for Diagnostic Classification of Dengue Fever Using Routine Hematological Parameters*. *Diagnostics*. 2026. https://pmc.ncbi.nlm.nih.gov/articles/PMC13605152/  
13. *Utilization of machine learning for dengue case screening*. *BMC Public Health*. 2024. https://pmc.ncbi.nlm.nih.gov/articles/PMC11167742/  
14. *A comparative analysis of dengue, chikungunya, and Zika manifestations in a pediatric cohort over 18 years*. 2025. https://pmc.ncbi.nlm.nih.gov/articles/PMC11741480.1/  
15. *Development and validation of an interpretable machine learning model for early hospital-based differentiation of chikungunya and dengue fever using routine clinical data*. *BMC Infectious Diseases*. 2026. https://www.link.springer.com/article/10.1186/s12879-026-13778-6  
16. *Integrating machine learning and time-to-event models to explain and predict risk of hospitalization due to dengue in Colombia*. *Scientific Reports*. 2025. https://link.springer.com/article/10.1038/s41598-025-22681-0  
17. *Risk Stratification of Dengue Cases Requiring Hospitalization*. *Journal of Medical Virology*. 2025. https://onlinelibrary.wiley.com/doi/10.1002/jmv.70511  
18. *Predictive Dynamics: Modeling for Virological Surveillance and Clinical Management of Dengue*. 2022. https://link.springer.com/chapter/10.1007/978-1-4614-3961-5_1  
19. *Decision Tree Algorithms Predict the Diagnosis and Outcome of Dengue Fever in the Americas*. *PLoS Neglected Tropical Diseases*. 2008. https://journals.plos.org/plosntds/article/file?id=10.1371/journal.pntd.0000196&type=printable  
20. *Comparison of Predictive Models for Severe Dengue*. 2024. https://pmc.ncbi.nlm.nih.gov/articles/PMC11793037/  
21. *Prognostic Models in Patients with Dengue*. 2025. https://pmc.ncbi.nlm.nih.gov/articles/PMC11965740/  
22. *Predictive markers for the early prognosis of dengue severity*. 2021. https://pmc.ncbi.nlm.nih.gov/articles/PMC8519480/  
23. *Assessing generalizability of a dengue classifier across populations*. 2025. https://pmc.ncbi.nlm.nih.gov/articles/PMC12132959/  
24. *A Systematic Review of Applications of Machine Learning and Other Soft Computing Techniques for the Diagnosis of Tropical Diseases*. 2022. https://pmc.ncbi.nlm.nih.gov/articles/PMC9787706/  
25. *Dengue models based on machine learning techniques: A systematic literature review*. *Artificial Intelligence in Medicine*. 2021. https://www.sciencedirect.com/science/article/pii/S0933365721001500  
26. *Diagnostic performance of artificial intelligence in early diagnosis of mosquito-borne viral infections: Systematic review and meta-analysis*. *PLOS Digital Health*. https://journals.plos.org/digitalhealth/article/file?id=10.1371/journal.pdig.0001796&type=printable  
27. *Dengue Prediction in Latin America Using Machine Learning and One Health Perspective*. 2022. https://pmc.ncbi.nlm.nih.gov/articles/PMC9611387/  
28. *Revisión sistemática de modelos predictivos para el dengue en Sudamérica*. 2026. https://sedici.unlp.edu.ar/handle/10915/191503  

### Estudios sobre comparación entre arbovirosis

29. *A comparative analysis of dengue, chikungunya, and Zika manifestations in a pediatric cohort over 18 years*. 2025. https://pmc.ncbi.nlm.nih.gov/articles/PMC11741480.2/  
30. *Simple Clinical and Laboratory Predictors of Chikungunya versus Dengue*. 2012. https://pmc.ncbi.nlm.nih.gov/articles/PMC3459852/  
31. *Arthralgia and fever as dominant predictors of Chikungunya*. 2026. https://pmc.ncbi.nlm.nih.gov/articles/PMC13110947/  

### Estudios sobre dinámica hematológica y gravedad

32. *Quantifying the kinetics of hematocrit and platelet count during dengue*. *PLoS Neglected Tropical Diseases*. https://journals.plos.org/plosntds/article/file?rev=2&id=10.1371/journal.pntd.0014245&type=printable  
33. *Clinical and Laboratory Predictors of Dengue Hemorrhagic Fever*. 2025. https://pmc.ncbi.nlm.nih.gov/articles/PMC12741299/  
34. *Utilities and Limitations of the World Health Organization 2009 Warning Signs for Adult Dengue Severity*. 2013. http://pmc.ncbi.nlm.nih.gov/articles/PMC3547865/  
35. *The Value of Warning Signs From the WHO 2009 Dengue Guidelines*. 2024. https://pmc.ncbi.nlm.nih.gov/articles/PMC11191040/  
36. *Predicting deterioration in dengue using a low cost wearable for continuous clinical monitoring*. *npj Digital Medicine*. 2024. https://link.springer.com/article/10.1038/s41746-024-01304-4  
37. *Development and evaluation of a diagnostic aiding tool for dengue using machine learning*. 2026. https://pmc.ncbi.nlm.nih.gov/articles/PMC12778211/  
38. *Performance of Dengue Diagnostic Tests in a Single-Specimen Diagnostic Algorithm*. *Journal of Infectious Diseases*. 2016. https://academic.oup.com/jid/article/214/6/836/2577307  
39. *A New Intelligence Based Approach for Computer-Aided Diagnosis of Dengue Fever*. 2011. https://ar5iv.labs.arxiv.org/html/1502.00062  

### Fuentes metodológicas sobre calibración

40. Collins GS, et al. *Evaluation of clinical prediction models (part 1): from development to external validation*. *BMJ*. 2024. https://www.bmj.com/content/384/bmj-2023-074819  
41. Collins GS, et al. *Evaluation of clinical prediction models (part 2): how to undertake an external validation study*. *BMJ*. 2024. https://www.bmj.com/content/384/bmj-2023-074820  
42. *Developing clinical prediction models: a step-by-step guide*. *BMJ*. 2024. https://www.bmj.com/content/386/bmj-2023-078276.full.pdf  
43. *Introduction to Clinical Prediction Models*. 2022. https://pmc.ncbi.nlm.nih.gov/articles/PMC10760493/  
44. *Appraising prediction research: a guide and meta-review*. 2021. https://pmc.ncbi.nlm.nih.gov/articles/PMC9291738/  
45. scikit-learn. *Probability calibration*. https://scikit-learn.org/1.8/modules/calibration.html  

### Nota sobre uso de fuentes

Las fuentes se organizan en cinco grupos:

- Guías y documentos normativos argentinos e internacionales.
- Artículos sobre modelos de machine learning para dengue.
- Estudios sobre comparación entre arbovirosis.
- Estudios sobre dinámica hematológica y gravedad.
- Fuentes metodológicas sobre calibración.

Esta organización permite respaldar cada sección del informe con evidencia específica y facilita la revisión por pares.
