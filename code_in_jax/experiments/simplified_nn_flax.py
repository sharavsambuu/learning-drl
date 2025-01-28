import flax.linen as nn
import jax
from jax import numpy as jnp
import numpy as np

class Dense(nn.Module):
    features: int
    kernel_init: callable = jax.nn.initializers.lecun_normal()
    bias_init: callable = jax.nn.initializers.zeros

    @nn.compact
    def __call__(self, x):
        kernel = self.param('kernel', self.kernel_init, (x.shape[-1], self.features))
        bias = self.param('bias', self.bias_init, (self.features,))
        return jnp.dot(x, kernel) + bias

# Неорон сүлжээний оролтын өгөгдлийг бэлтгэх
# энд 1 ширхэг дата буюу batch нь 1 гэсэн үг 4-н оролтын өгөгдлийн дүрс
x = jnp.ones((1, 4))

# Оролтын өгөгдлийн дүрсээс хамааруулан неорон сүлжээний жингүүдийг
# эхний байдлаар цэнэглэн гаргаж авах
module = Dense(features=4)
params = module.init(jax.random.PRNGKey(0), x)['params']
print("Неорон сүлжээний эхний цэнэглэгдсэн параметрүүд :")
print(params)
y = module.apply({'params': params}, x)
print("Эхний байдлаар predict хийсэн гаралт :")
print(y)

# Цэнэглэгдсэн параметрүүдээ хэрэглэн энгийн predict хийх forward дуудалт
prediction = module.apply({'params': params}, x)
print("predict хийсэн дуудалт :")
print(prediction)

# Зөвхөн hyperparameter-үүд тохируулсан модуль үүсгэе гэвэл nn.compact ашиглаж болно
# Модуль хэрэглэн forward дуудалт хийх
prediction = module.apply({'params': params}, x)
print("Модулиас хийсэн prediction дуудалтын үр дүн :")
print(prediction)

# Модуль ашигласнаар хооронд нь хольж бүрдүүлж болдог нарийн
# бүтэцтэй неорон сүлжээ зохион байгуулах боломж бүрддэг

def relu(x):
    return jnp.maximum(0., x)

# Multi Layer Perceptron буюу олон давхаргатай хялбар неорон сүлжээ
class MLP(nn.Module):
    hidden_features: int
    output_features: int
    activation_fn: callable

    @nn.compact
    def __call__(self, x):
        x = Dense(features=self.hidden_features)(x)
        x = self.activation_fn(x)
        x = Dense(features=self.output_features)(x)
        return x

module = MLP(hidden_features=8, output_features=4, activation_fn=relu)
params = module.init(jax.random.PRNGKey(0), x)['params']
# Нарийн бүтэцтэй олон давхаргатай неорон сүлжээ бий болгосон учраас
# параметрүүд нь шаталсан бүтэцтэй dictionary байна
print("MLP параметрүүд :")
print(params)
y = module.apply({'params': params}, x)
print("MLP predict хийсэн утгууд :")
print(y)

# Хэрвээ неорон давхаргад нэр оноогоогүй байвал автоматаар дугаарлагдаад явна
# тогтсон нэр өгч ашиглах боломжтой

class NamedMLP(nn.Module):
    hidden_features: int
    output_features: int
    activation_fn: callable

    @nn.compact
    def __call__(self, x):
        x = Dense(features=self.hidden_features, name='hidden')(x)
        x = self.activation_fn(x)
        x = Dense(features=self.output_features, name='out')(x)
        return x

module = NamedMLP(hidden_features=8, output_features=4, activation_fn=relu)
params = module.init(jax.random.PRNGKey(0), x)['params']

print("Нэрлэсэн давхаргууд бүхий неорон сүлжээний бүтэц :")
print(jax.tree_util.tree_map(np.shape, params))

# Олон call дуудалтанд дундын parameter-үүд хэрэглэж болно

class SimpleRNN(nn.Module):
    features: int
    iterations: int = 3

    @nn.compact
    def __call__(self, x):
        dense = Dense(features=self.features,
                      kernel_init=jax.nn.initializers.orthogonal(), name='cell')
        ys = []
        for i in range(self.iterations):
            x = dense(x)
            ys.append(x)
        return ys

# Оролтын өгөгдлийн дүрснээс хамаарч модулийг init хийх нь заримдаа шаардлагагүй нэмэлт
# үйлдлүүд хийгдэх магадлалтай, зөвхөн оролтын дүрс нь ямар вэ гэдгээр init хийж болно
# {'cell': {'bias': (2,), 'kernel': (2, 2)}}

# Corrected initialization:
module = SimpleRNN(features=2)  # Use correct features here
input_shape = jnp.zeros((1, 2))  # Input shape for shape inference
variables = module.init(jax.random.PRNGKey(0), input_shape) # Initialize only once with input_shape
params = variables['params']
print("Shape inference горимоор барьсан неорон сүлжээний бүтэц :")
print(jax.tree_util.tree_map(np.shape, params))

# Apply with an input that has a shape compatible with the shape inference
compatible_x = jnp.ones((1, 2))
ys = module.apply({'params': params}, compatible_x)
print("Хялбар Recurrent Neural Network-ийн гаралт :")
print(ys)

# cell давхаргыг 3-н удаа дуудсанч shared параметер горимоор үүсгээн тул нэмэлтээр
# 3-н давхарга болж орж ирэхгүй нэг л давхарга байдлаар харагдана гэсэн үг
# {'cell': {'bias': (4,), 'kernel': (4, 4)}}
print("Recurrent Neural Network-ийн параметер бүтэц :")
print(jax.tree_util.tree_map(np.shape, params))

# Модуль хэрэглэснээр неорон сүлжээний параметрүүдийг init болон apply функцын тусламжтайгаар
# ил хэлбэрээр track хийх боломж олгодог.
x = jnp.ones((1, 2))
module = Dense(features=4)
variables = module.init(jax.random.PRNGKey(0), x)
params = variables['params']
model = variables

print("Моделийн бүтцийг харах :")
print(jax.tree_util.tree_map(np.shape, model['params']))

prediction = module.apply(model, x)
print("Моделиор predict хийсэн үр дүн :")
print(prediction)

print("Моделийн параметрүүд :")
print(model['params'])

# Моделийн параметрүүдийг нь шинэчилж сольж болно
biased_model = {
    'params': {
        'kernel': model['params']['kernel'],
        'bias': model['params']['bias'] + 1.
    }
}

print("Параметрүүдийг нь шинэчилсэн моделийн параметрүүд :")
print(biased_model['params'])

# Модель нь JAX-ны pytree объект байдлаар зохион байгуулагддаг тул
# JAX-н native хувиргалтуудад асуудалгүйгээр хэрэглэх боломжтой

# Жишээлбэл модель объект хэрэглэн градиент утгууд олж болно
def loss_fn(params):
    y = module.apply({'params': params}, x)
    return jnp.mean(jnp.square(y))

model_gradients = jax.grad(loss_fn)(model['params'])
print("Моделийн loss-оос авсан gradient утгууд")
print(model_gradients)

# BatchNorm мэтийн тохиолдолд batch-уудын mean, variance гэх мэт статистик өгөгдлүүдийг
# хадгалаж хэрэглэх хэрэгцээ гардаг энэ тохиолдолд Module.state-г хэрэглэж болно

class BatchNorm(nn.Module):
    red_axis: int = 0
    eps: float = 1e-5
    momentum: float = 0.99
    gamma_init: callable = jax.nn.initializers.ones
    beta_init: callable = jax.nn.initializers.zeros

    @nn.compact
    def __call__(self, x, training: bool):
        # Оролтын моментүүдийг тооцоолох
        mean = x.mean(self.red_axis, keepdims=True)
        var = jnp.square(x - mean).mean(self.red_axis, keepdims=True)
        # State хувьсагчууд тодорхойлох
        ra_mean = self.variable('stats', 'mean', lambda shape=mean.shape, dtype=jnp.float32: jax.nn.initializers.zeros(self.make_rng('stats'), shape, dtype))
        ra_var = self.variable('stats', 'var', lambda shape=var.shape, dtype=jnp.float32: jax.nn.initializers.ones(self.make_rng('stats'), shape, dtype))
        # Жингүүдийг эхний байдлаар цэнэглэж байгаа бол
        # average утгуудыг тооцох шаардлага байхгүй
        if not self.is_initializing():
            if training:
                # Сургаж байгаа үед average-үүдийг шинэчлэх ёстой
                alpha = 1. - self.momentum
                ra_mean.value = alpha * (mean - ra_mean.value) + ra_mean.value
                ra_var.value = alpha * (var - ra_var.value) + ra_var.value
            else:
                # Сургаагүй тохиолдолд average-уудаа хэрэглэнэ
                mean = ra_mean.value
                var = ra_var.value
        # Оролтын өгөгдлөө стандартчилах
        y = (x - mean) / jnp.sqrt(var + self.eps)
        # Оролтын scale болон bias-уудыг суралцах
        gamma = self.param('gamma', self.gamma_init, mean.shape)
        beta = self.param('beta', self.beta_init, mean.shape)
        return gamma * y + beta

# Хэрэв state хэрэглэхээр бол flax.linen.Module-ийн variable-ийг ашиглана
# Энэ state-үүд 'stats' collection дотор хадгалагддаг

class MyModel(nn.Module):
    @nn.compact
    def __call__(self, x, training=False):
        x = Dense(features=4)(x)
        x = BatchNorm(momentum=0., name='batch_norm')(x, training=training)
        return x

dist_a = lambda rng, shape: jax.random.normal(rng, shape) * jnp.array([[1., 3.]])
x_a = dist_a(jax.random.PRNGKey(1), (1024, 2))
print("Оролтын өгөгдлын стандарт хазайлт :", x_a.std(0))

variables = MyModel().init(jax.random.PRNGKey(2), x_a, training=True)
params, stats = variables['params'], variables['stats']
y = MyModel().apply(variables, x_a, training=False, mutable=['stats'])
print("Гаралтын стандарт хазайлт (init) :", y[0].std(0))

y, mutated_state = MyModel().apply(variables, x_a, training=True, mutable=['stats'])
print("Гаралтын стандарт хазайлт (training) :", y.std(0))

y = MyModel().apply({'params': params, 'stats': mutated_state['stats']}, x_a, training=False, mutable=['stats'])
print("Гаралтын стандарт хазайлт (testing) :", y[0].std(0))

# state өгөгдлийг Collection.as_dict() ээр шалгаж болно
print("Эхлэл state :")
print(variables['stats'])

print("Шинэ state :")
print(mutated_state['stats'])

# state тооцох механизм нь ил байх ёстой жишээлбэл
# тест хийж байх үед state тооцоолох шаардлагагүй
# мөн өөр оролтын өгөгдлүүд хэрэглэн өөр статистик
# state цуглуулах шаардлага гардаг энэ тохиолд
# ил байдлаар state тооцохын үр ашиг гардаг
dist_b = lambda rng, shape: jax.random.normal(rng, shape) * jnp.array([[2., 5.]])
x_b = dist_b(jax.random.PRNGKey(1), (1024, 2))
y = MyModel().apply({'params': params, 'stats': mutated_state['stats']}, x_b, training=False, mutable=['stats'])
print(y[0].std(0))  # Энэ тохиолдолд зөв нормчилогдож чадахгүй

y, state_b = MyModel().apply(variables, x_b, training=True, mutable=['stats'])
print("Гаралтын стандарт хазайлт (training) :", y.std(0))

y = MyModel().apply({'params': params, 'stats': state_b['stats']}, x_b, training=False, mutable=['stats'])
print("Гаралтын стандарт хазайлт (testing) :", y[0].std(0)) 