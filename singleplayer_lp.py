import numpy as np 
import converters

# Integer LP instance:
# tuple of (R, A, b)

# want to maximize (R dot x)
# subject to Ax <= b



def gen_instance(convs, resources):
    #todo: add that unused resources are worth points
    convs = sorted(set(convs))
    reward = np.array([c.output for c in convs])

    max_usage = np.array([resources[t] for t in converters.resource_types] + #don't overspend resources.
                [1 for c in convs] + # use each converter at most once
                [0 for c in convs]) #nonnegativity
    
    machine_usage = np.matrix([[c.inputs.count(t) for c in convs] for t in converters.resource_types] + 
        [[1 if c0 == c else 0 for c0 in convs] for c in convs] +
        [[-1 if c == c0 else 0 for c in convs] for c0 in convs])

    return (reward, machine_usage, max_usage)

if __name__ == '__main__':
    import instance_gen
    convs = instance_gen.gen_converters(7,7,7)
    ress = instance_gen.gen_resources(10)

    print(gen_instance(convs,ress))
