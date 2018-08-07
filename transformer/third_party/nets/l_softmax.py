import numpy as np
import math
import uuid
import tensorflow as tf

margin = 4
beta = 100
scale = 0.99
beta_min = 0
eps = 0
c_map = []
k_map = []
c_m_n = lambda m, n: math.factorial(n) / math.factorial(m) / math.factorial(n - m)
for i in range(margin + 1):
    c_map.append(c_m_n(i, margin))
    k_map.append(math.cos(i * math.pi / margin))


def find_k(cos_t):
    '''find k for cos(theta)
    '''
    # for numeric issue
    eps = 1e-5
    le = lambda x, y: x < y or abs(x - y) < eps
    for i in range(margin):
        if le(k_map[i + 1], cos_t) and le(cos_t, k_map[i]):
            return i
    raise ValueError('can not find k for cos_t = %f' % cos_t)


def find_k_vector(cos_t_vec):
    k_val = []
    for i in range(cos_t_vec.shape[0]):
        try:
            k_val.append(find_k(cos_t_vec[i]))
        except ValueError:
            print(cos_t_vec)
    return k_val


def calc_cos_mt(cos_t):
    '''calculate cos(m*theta)
    '''
    cos_mt = 0
    sin2_t = 1 - cos_t * cos_t
    flag = -1
    for p in range(margin // 2 + 1):
        flag *= -1
        cos_mt += flag * c_map[2*p] * pow(cos_t, margin-2*p) * pow(sin2_t, p)
    return cos_mt


def calc_cos_mt_vector(cos_t_vector):
    cos_mt_val = []
    for i in range(cos_t_vector.shape[0]):
        cos_mt_val.append(calc_cos_mt(cos_t_vector[i]))
    return cos_mt_val


def lsoftmax(x, weights, labels):
    def _lsoftmax(net_val, weights, labels):
        global beta, scale
        normalize_net = np.linalg.norm(net_val, axis=1).reshape([net_val.shape[0], 1])
        normalize_weights = np.linalg.norm(weights, axis=0).reshape([-1, weights.shape[1]])
        normalize_val = normalize_net * normalize_weights

        indexes = np.arange(net_val.shape[0])
        labels = labels.reshape((-1,))

        normalize_val_target = normalize_val[indexes, labels]
        logit = np.dot(net_val, weights)
        cos_t_target = logit[indexes, labels] / (normalize_val_target + eps)
        k_val = np.array(find_k_vector(cos_t_target))
        cos_mt_val = np.array(calc_cos_mt_vector(cos_t_target))
        logit_output_cos = np.power(-1, k_val) * cos_mt_val - 2 * k_val
        logit_output = logit_output_cos * normalize_val_target
        logit_output_beta = (logit_output + beta * logit[indexes, labels]) / (1 + beta)
        logit[indexes, labels] = logit_output_beta
        return logit

    def _lsoftmax_grad(x, w, label, grad):
        global beta, scale, beta_min
        # original without lsoftmax
        w_grad = x.T.dot(grad)   # 2, 10
        x_grad = grad.dot(w.T)  # 2, 2
        n = label.shape[0]
        m = w.shape[1]
        feature_dim = w.shape[0]
        cos_t = np.zeros(n, dtype=np.float32)
        cos_mt = np.zeros(n, dtype=np.float32)
        sin2_t = np.zeros(n, dtype=np.float32)
        fo = np.zeros(n, dtype=np.float32)
        k = np.zeros(n, dtype=np.int32)
        x_norm = np.linalg.norm(x, axis=1)
        w_norm = np.linalg.norm(w, axis=0)
        w_tmp = w.T
        for i in range(n):
            yi = int(label[i])
            f = w_tmp[yi].dot(x[i])
            cos_t[i] = f / (w_norm[yi] * x_norm[i])
            k[i] = find_k(cos_t[i])
            cos_mt[i] = calc_cos_mt(cos_t[i])
            sin2_t[i] = 1 - cos_t[i]*cos_t[i]
            fo[i] = f
        # gradient w.r.t. x_i
        for i in range(n):
            # df / dx at x = x_i, w = w_yi
            j = yi = int(label[i])
            dcos_dx = w_tmp[yi] / (w_norm[yi]*x_norm[i]) - x[i] * fo[i] / (w_norm[yi]*pow(x_norm[i], 3))
            dsin2_dx = -2 * cos_t[i] * dcos_dx
            dcosm_dx = margin*pow(cos_t[i], margin-1) * dcos_dx  # p = 0
            flag = 1
            for p in range(1, margin//2+1):
                flag *= -1
                dcosm_dx += flag * c_map[2*p] * (p*pow(cos_t[i], margin-2*p)*pow(sin2_t[i], p-1)*dsin2_dx +
                                                 (margin-2*p)*pow(cos_t[i], margin-2*p-1)*pow(sin2_t[i], p)*dcos_dx)
            df_dx = (pow(-1, k[i]) * cos_mt[i] - 2*k[i]) * w_norm[yi] / x_norm[i] * x[i] + \
                     pow(-1, k[i]) * w_norm[yi] * x_norm[i] * dcosm_dx
            alpha = 1 / (1 + beta)
            x_grad[i] += alpha * grad[i, yi] * (df_dx - w_tmp[yi])
        # gradient w.r.t. w_j
        for j in range(m):
            dw = np.zeros(feature_dim, dtype=np.float32)
            for i in range(n):
                yi = int(label[i])
                if yi == j:
                    # df / dw at x = x_i, w = w_yi and yi == j
                    dcos_dw = x[i] / (w_norm[yi]*x_norm[i]) - w_tmp[yi] * fo[i] / (x_norm[i]*pow(w_norm[yi], 3))
                    dsin2_dw = -2 * cos_t[i] * dcos_dw
                    dcosm_dw = margin*pow(cos_t[i], margin-1) * dcos_dw  # p = 0
                    flag = 1
                    for p in range(1, margin//2+1):
                        flag *= -1
                        dcosm_dw += flag * c_map[2*p] * (p*pow(cos_t[i], margin-2*p)*pow(sin2_t[i], p-1)*dsin2_dw +
                                                (margin-2*p)*pow(cos_t[i], margin-2*p-1)*pow(sin2_t[i], p)*dcos_dw)
                    df_dw_j = (pow(-1, k[i]) * cos_mt[i] - 2*k[i]) * x_norm[i] / w_norm[yi] * w_tmp[yi] + \
                               pow(-1, k[i]) * w_norm[yi] * x_norm[i] * dcosm_dw
                    dw += grad[i, yi] * (df_dw_j - x[i])
            alpha = 1 / (1 + beta)
            w_grad[:, j] += alpha * dw
        beta *= scale
        beta = max(beta, beta_min)
        return x_grad, w_grad

    def _lsoftmax_grad_op(op, grad):
        x = op.inputs[0]
        weights = op.inputs[1]
        labels = op.inputs[2]
        x_grad, w_grad = tf.py_func(_lsoftmax_grad, [x, weights, labels, grad], [tf.float32, tf.float32])
        return x_grad, w_grad, labels

    grad_name = 'lsoftmax_' + str(uuid.uuid4())
    tf.RegisterGradient(grad_name)(_lsoftmax_grad_op)

    g = tf.get_default_graph()
    with g.gradient_override_map({"PyFunc": grad_name}):
        output = tf.py_func(_lsoftmax, [x, weights, labels], tf.float32)
    return output