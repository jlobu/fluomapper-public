from argparse import Namespace

import numpy as np
from joblib import Parallel, delayed
import itertools
import inspect
import argparse

import re

from test_tube.argparse_hopt import TTNamespace
import copy


def _remove(lis, el):
    if not type(el) in (list, tuple):
        el = [el]

    for e in el:
        if e in lis:
            lis.remove(e)

    return lis


def _add(lis, el):
    if not type(el) in (list, tuple):
        el = [el]
    
    for e in el:
        if e not in lis:
            lis += [e]

    return lis


def glob_re(pattern, strings):
    return filter(re.compile(pattern).match, strings)


def run_jobs(jobs, joblib=True, n_jobs=4, chunks=1, chunk_callback=None, *args, **kwargs):
    if len(jobs) == 0:
        return None, None

    if joblib:
        jobs = [delayed(job)() for job in jobs]

        chunk_size = max(1, len(jobs) // chunks)
        chunks = [jobs[i:i + chunk_size] for i in range(0, len(jobs), chunk_size)]

        out = []
        for chunk in chunks:
            chunk_out = Parallel(n_jobs=n_jobs, *args, **kwargs)(chunk)
            if chunk_callback is not None:
                ret = chunk_callback(chunk_out, args=[job[0].args for job in chunk],
                                     kwargs=[job[0].keywords for job in chunk])
                out.append((chunk_out, ret))

            else:
                out.append(chunk_out)
    else:
        out = []
        # create chunks
        nr_chunks = chunks
        chunk_size = max(1, len(jobs) // chunks)
        chunks = [jobs[i:i + chunk_size] for i in range(0, len(jobs), chunk_size)]

        for j, chunk in enumerate(chunks):
            chunk_out = []

            for i, job in enumerate(chunk):
                if 'verbose' in kwargs and kwargs['verbose']:
                    print('\r\r Chunk %d / %d' % (j, nr_chunks) +
                          '\n Working on job %d/%d, ' % (i, len(chunk)) +
                          '\n args: %s, \n dset_kwargs: %s' % (', '.join(job.args), ', '.join([str(tup)
                                                                                          for tup in job.keywords.items()])))
                chunk_out.append(job())

            if chunk_callback is not None:
                ret = chunk_callback(chunk_out, args=[job.args for job in chunk],
                                     kwargs=[job.keywords for job in chunk])
                out.append((chunk_out, ret))

            else:
                out.append(chunk_out)

    return list(itertools.chain(*out))


def chunk_list(it, size):
    it = iter(it)
    return list(iter(lambda: tuple(itertools.islice(it, size)), ()))


def init_with_valid_kwargs(cls, *args, **kwargs):
    valid_params = list(inspect.signature(cls.__init__).parameters)
    valid_kwargs = dict([(k, v) for k, v in kwargs.items() if k in valid_params])

    return cls(*args, **valid_kwargs)


def eval_with_valid_kwargs(obj, method, *args, **kwargs):
    if obj is not None:
        valid_params = list(inspect.signature(getattr(obj, method)).parameters)
        valid_kwargs = dict([(k, v) for k, v in kwargs.items() if k in valid_params])
        return getattr(obj, method)(*args, **valid_kwargs)

    else:
        valid_params = list(inspect.signature(method).parameters)
        valid_kwargs = dict([(k, v) for k, v in kwargs.items() if k in valid_params])
        return method(*args, **valid_kwargs)


def add_model_specific_args(parser, parser_spec, prepend='', ignore_overrides=False, default_none=False, **kwargs):
    if prepend is None:
        prepend = ''

    for key, val in parser_spec.items():
        if default_none:
            val['default'] = None

        if prepend != '' and not prepend.endswith('.'):
            prepend = '%s.' % prepend

        try:
            parser.add_argument('--%s%s' % (prepend, key), **val)

        except argparse.ArgumentError as e:
            if ignore_overrides:
                pass

            else:
                raise e

    return parser


def add_prepend(prepend, prepend2):
    if prepend is None or prepend == '':
        return prepend2

    if prepend.endswith('.'):
        prepend = prepend[:-1]

    return f'{prepend}.{prepend2}'


def _stack_dict(dic, keys):
    if keys is None:
        return

    if type(keys) is str:
        keys = [keys]

    ret = np.stack([dic[key] for key in keys], axis=0)
    return ret


def update_namespace(namespace1, namespace2, return_ttnamespace=False, ignore_none=False):

    if namespace2 is None:
        return namespace1

    if namespace1 is None:
        return namespace2

    if type(namespace1) in (Namespace, TTNamespace):
        return_ttnamespace = type(namespace1) is TTNamespace
        namespace1 = vars(namespace1)

    if type(namespace2) in (Namespace, TTNamespace):
        return_ttnamespace = return_ttnamespace or type(namespace1) is TTNamespace
        namespace2 = vars(namespace2)

    if ignore_none:
        namespace2 = {k : v for k, v in namespace2.items() if v is not None}

    namespace1.update(namespace2)

    if return_ttnamespace:
        return TTNamespace(**namespace1)

    return Namespace(**namespace1)


def make_parser_soft(parser):
    for arg in parser._actions:
        arg.required = False
        arg.default = None

    return parser


def dict_compare(d1, d2):
    d1_keys = set(d1.keys())
    d2_keys = set(d2.keys())
    shared_keys = d1_keys.intersection(d2_keys)
    added = d1_keys - d2_keys
    removed = d2_keys - d1_keys
    modified = {o : (d1[o], d2[o]) for o in shared_keys if d1[o] != d2[o]}
    same = set(o for o in shared_keys if d1[o] == d2[o])
    return added, removed, modified, same


def split_kwargs(dic, key, update_dic=True):
    new_dic = dict([('.'.join(k.split('.')[1:]), val) for k, val in dic.items()
                    if len(k.split('.')) > 1 and k.split('.')[0] == key])
    
    if update_dic:
        ret_dic = copy.deepcopy(dic)
        ret_dic.update(new_dic)
        return ret_dic

    return new_dic

def make_iterable(obj):
    if not isinstance(obj, (list, tuple)):
        obj = [obj]
    
    return obj

