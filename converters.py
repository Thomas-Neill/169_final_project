from collections import namedtuple
 
resource_types = 'OURBGYW'

Converter = namedtuple('Converter', ['name', 'inputs', 'output'])

t1_converters = [
    Converter('Atomic Transmutation', 'O', 4),
    Converter('Clinical Immortality', 'UU', 7),
    Converter('Genetic Engineering', 'GGG', 4),
    Converter('Nanotechnology', 'RRR', 4),
    Converter('Quantum Computing', 'BB', 4),
    Converter('Ubiquitous Cultural Repository', 'YY', 7),
    Converter('Universal Translator', 'WWW', 7)]

t2_converters =    [
    Converter('Achronal Analysis', 'BBB', 6.5),
    Converter('Antimatter Power', 'RRRR', 5.5),
    Converter('Cross Species Ethical Equality', 'WWWW', 10),
    Converter('Hyperspace Mining', 'YYY', 7.5),
    Converter('Interspecies Medical Exchange', 'GGGG', 9),
    Converter('Organic Construction', 'UUU', 6),
    Converter('Singularity Control', 'OO', 8.5)]

t3_converters = [
    Converter('Galactic Telecom Control', 'BBBB', 21),
    Converter('Matter Generation', 'YYYY', 18.5),
    Converter('Megastructures', 'OOO', 32),
    Converter('Poly Species Corporations', 'GGGGG', 15),
    Converter('Social Exodus', 'WWWWW', 20.5),
    Converter('Temporal Dilation', 'UUUU', 18),
    Converter('Xeno Cultural Exchange', 'RRRRR', 17.5)]
