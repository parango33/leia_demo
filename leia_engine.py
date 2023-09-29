import os
from typing import Any
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain import PromptTemplate
from langchain.chains import LLMChain
from output_parsers import document_intel_parser, LegalDocumentIntel


load_dotenv()


############## PODER ################
ejemplo_poder_v2="""
Señores Dirigido
Asunto: xxxxxx
Respetados señores:
Yo, CLIENTE, identificado con cédula de ciudadanía número [Número de documento de identificación], por medio del presente documento, otorgo poder especial, amplio y suficiente al abogado ABOGADO, 
identificado con cédula de ciudadanía número [Número de documento de identificación], 
para que me represente y actúe en mi nombre dentro del radicado [Número de radicado], ante la Fiscalía General de la Nación, a partir de la fecha.
Además de las facultades inherentes al presente mandato, el abogado ABOGADO queda revestido de todas las facultades que le otorga la ley, en especial, 
las de conciliar, recibir, transigir, desistir, sustituir, reasumir, asignar apoderados suplentes, interponer recursos y, en general, desplegar toda actividad tendiente al cumplimiento de su gestión como profesional del derecho en relación con el mencionado proceso.

Dejo constancia de que este poder se otorga de manera voluntaria y consciente, confiando plenamente en la capacidad y experiencia del abogado ABOGADO para representarme y defender mis intereses legales en el proceso mencionado.
Agradezco a la Fiscalía General de la Nación su atención a este poder y solicito que se reconozca y acepte la representación del abogado ABOGADO en el presente caso.
Sin otro particular, quedo a disposición para cualquier comunicación o trámite adicional que requieran.
Atentamente,
CLIENTE Cédula de ciudadanía: [Número de documento de identificación]
Acepto
ABOGADO Cédula de ciudadanía: [Número de documento de identificación] Abogado

"""
ejemplo_poder="""
Yo CLIENTE, mayor y domiciliado en ____________, identificado como aparece al pie
de mi firma, le manifiesto por medio del presente escrito que le confiero poder especial al señor
ABOGADO, también mayor y vecino de ________, identificado con la CC No. __________expedida
en _____________, para presentar ante su Despacho solicitud para (se menciona la clase de
actuación solicitada), realizada en un inmueble ubicado en la (dirección) de al ciudad ______, identificado además con la referencia catastral N° _______________ .

Nuestro apoderado queda facultado para Solicitar, tramitar, notificarse retirar y demás facultades que
sean necesarias para el cumplimiento de su mandato y obtener la expedición del Acto Administrativo
expedido por su Despacho.

Solicitamos con todo respeto sea reconocido nuestro apoderado en los términos y para los efectos del
presente poder.

Atentamente,



FIRMA
NOMBRE
CC No. _____________de _______________
PODERDANTE


Acepto, """
apoderado="Miguel Sanchez"
abogado="Pedro Perez"
fecha="2023-04-21"
objetivo_poder=f"Este documento se dirige a la fiscalia colombiana y debe decir que {apoderado} otorga poder especial amplio y suficiente a {abogado}"


############# TUTELA #################
ejemplo_tutela = """
Ref . 	ACCIÓN DE TUTELA de {nombre} contra {demandado}
I. HECHOS

{hechos}
La presente solicitud de tutela se apoya en los siguientes hechos:

1. 

2. 

3. 


II. DERECHOS VULNERADOS 

{derecho_vulnerado}

III. FUNDAMENTOS DE DERECHO 

El derecho a la educación es un derecho constitucional y un servicio público esencial a cargo del Estado y en favor de todos los habitantes del territorio nacional. 

Por otra parte de conformidad con la sentencia T- 348 de 2007, la maternidad es una decisión de la mujer de traer al mundo una nueva vida, razón por la cual es una arista del núcleo esencial del derecho fundamental al libre desarrollo de la personalidad y por ende, no puede ser objeto de injerencia por autoridad pública o por particular alguno. En este sentido, se consideran contrarias a los postulados constitucionales todas aquellas medidas que tiendan a impedir o a hacer más gravoso el ejercicio de la mencionada opción vital.

Asimismo, el embarazo de una estudiante no es una situación que deba ser limitada o restringida, claramente ni los manuales de convivencia de las instituciones educativas, ni el reglamento interno, pueden, ni explícita, ni implícitamente, tipificar como falta o causal de mala conducta, el embarazo de una estudiante. En efecto la Corte Constitucional ha establecido que toda norma reglamentaria que se ocupe de regular la maternidad en el sentido antes indicado debe ser inaplicada por los jueces constitucionales, por ser contraria a la Carta Política.
 
En conclusión, la negación por parte de (Entidad(es) que está vulnerando el derecho a la educación, es decir Colegio, escuela, universidad, entre otros) a realizar (Escriba las acciones que debe hacer la Entidad(es) para que cese la vulneración del derecho a la educación. Ejemplo: permitir el ingreso a clases o matricula de la estudiante sin importar su estado de embarazo, acciones positivas para evitar la discriminación por el estado de embarazo, disfrute de la licencia de maternidad, entre otras), son violaciones evidentes al derecho fundamental al libre desarrollo de la personalidad y a la educación.

IV. PETICIÓN DE TUTELA

(Deberá detallar de una forma clara, concreta y por separado, las acciones que solicita por parte de la Entidad(es) que está vulnerando los derechos a la educación y a la igualdad, es decir Colegio, escuela, universidad, entre otros. Ejemplo: no permitir el ingreso a clases o matricula por el estado de embarazo, acciones discriminatorias por el estado de embarazo, no disfrute de la licencia de maternidad, entre otros)

Con la presente ACCIÓN DE TUTELA se pretende:

1º. Que se tutele el derecho a la EDUCACIÓN, a la IGUALDAD, a la INTIMIDAD y al LIBRE DESARROLLO DE LA PERSONALIDAD de la accionante (o de la menor de edad (nombre del menor de edad)) 

2º. Que se ordene a (Entidad(es) que está vulnerando el derecho a la educación, es decir Colegio, escuela, universidad, entre otros) realizar las siguientes acciones (describa las acciones que se deben realizar para que cese la vulneración, tales como: permitir el ingreso a clases o matricula, disfrute de la licencia de maternidad, entre otros.) dentro de las cuarenta y ocho (48) horas siguientes a la sentencia de tutela.
"""

print("Procesando consulta...")

def detect_document(doc_indicator):
    my_dict = {
        "poder": ejemplo_poder_v2,
        "tutela": ejemplo_tutela
    }

    if doc_indicator in my_dict:
        return my_dict[doc_indicator]
    else:
        return "Documento no disponible"



def parse_document(doc_indicator, objective, n1,n2) -> Any:

    print("Procesando documento...")

    doc_ex = detect_document(doc_indicator)
    cliente = n1
    abogado = n2

    summary_template = """
    Dado la estructura de {ejemplo_documento} de un documento legal  y un objetivo {objective} quiero que me des:
    1. resumen corto del objetivo. Maximo 50 tokens.
    2. Un documento similar al ejemplo. Modifique el documento para ajustarse al nombre del cliente {apoderado} y el abogado {abogado}. Convierte el documento en uno más largo y más detallado que el original. Incluye el objetivo en el documento. 
    Limite su respuesta al maximo de caracteres que puede procesar.
    \n{format_instructions} 
    """

    summary_prompt_template = PromptTemplate(
         input_variables=["ejemplo_documento","objective", "apoderado","abogado"], 
         template=summary_template,
         partial_variables={"format_instructions": document_intel_parser.get_format_instructions()}
         )

    llm = ChatOpenAI(temperature=1, model_name="gpt-3.5-turbo")

    chain = LLMChain(llm=llm, prompt=summary_prompt_template)
    result = chain.run(ejemplo_documento=doc_ex, apoderado=n1, abogado=n2, objective=objective)
    print(result)
    print(document_intel_parser.parse(result))
    return document_intel_parser.parse(result)
    #print(result)
    #return result


if __name__ == "__main__":
    print(objetivo_poder)
    parse_document(doc_indicator="poder",objective=objetivo_poder,n1=apoderado,n2=abogado)




