import json
import os

EOS_TOKEN = "<|endoftext|>"

programs = [
"""enchufar Chamuyo

el doble dado x da 2# * x
el dosVeces dados f x da f (f x)
el cuádruple es dosVeces doble \n<|end|>""",

"""enchufar Chamuyo

la fruta es "Manzana"
el tres es 1# + 2#
los numeritos son [1#, 2#, 3#, 4#] \n<|end|>""",


"""enchufar Chamuyo   

el programa es
    escupir(mostrar(factorial(7#)));   

el factorial de Numerito en Numerito
  dado 0# da 1#
  dado n  da n * factorial (n - 1#) \n<|end|>""",

"""enchufar Chamuyo

el neg de Posta en Posta
  dado Sí da No
  dado No da Sí                      

el y de Posta en Posta en Posta
  dados Sí Sí da Sí
  dados Sí No da No
  dados No Sí da No
  dados No No da No                  

el o de Posta en Posta en Posta
  dados Sí Sí da Sí
  dados Sí No da Sí
  dados No Sí da Sí
  dados No No da No  \n<|end|>""",

"""enchufar Chamuyo

el programa es
    escupir(mostrar(neg Sí)) ;            
    escupir(mostrar(y No Sí)) ;           
    escupir(mostrar(o (neg Sí) Sí))   

el neg de Posta en Posta
  dado Sí da No
  dado No da Sí                      

el y de Posta en Posta en Posta
  dados Sí Sí da Sí
  dados Sí No da No
  dados No Sí da No
  dados No No da No                  

el o de Posta en Posta en Posta
  dados Sí Sí da Sí
  dados Sí No da Sí
  dados No Sí da Sí
  dados No No da No  \n<|end|>""",

"""enchufar Chamuyo

el mínimo                  
  dada [] da 0#
  dada [x] da x
  dada (x : y : xs)
    si x < y da mínimo (x : xs)
    si no    da mínimo (y : xs) \n<|end|>""",

"""enchufar Chamuyo

el programa es
   escupir(mostrar(numerosHasta 10#))

los numerosHasta
  dado 0# da []
  dado n  da numerosHasta (n - 1#) ++ [n]  \n<|end|>""",
  
"""enchufar Chamuyo

OJO. Composición de funciones.
la composición de (mengano en zutano)
               en (fulano en mengano)
               en fulano en zutano
  dadas f g da
    la que dado x da f (g x)

OJO. Aplicar una función a cada elemento de una lista.
el apl de (coso en cosito) en [coso] en [cosito]
  dadas _ []       da []
  dadas f (x : xs) da f x : apl f xs

OJO. Dar vuelta el orden de los argumentos.
el vueltargar de (a en b en c) en b en a en c
  dadas f x y da f y x  \n<|end|>""" 
  

]

output_path = 'data/dataset.jsonl'
os.makedirs('data', exist_ok=True)

EOS_TOKEN = "<|endoftext|>"

with open(output_path, 'w', encoding='utf-8') as f:
    for p in programs:
        # json.dumps handles the \n and quotes automatically
        entry = {"text": p.strip()}
        f.write(json.dumps(entry) + '\n')

print(f"✅ Created {output_path} with {len(programs)} samples.")