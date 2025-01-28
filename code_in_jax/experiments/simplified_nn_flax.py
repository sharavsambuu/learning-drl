import flax
import jax
from jax import numpy as jnp
import numpy as np

# Flax сурах дадлага : https://flax.readthedocs.io/en/latest/notebooks/flax_guided_tour.html


# Flax ашиглан хамгийн хялбар неорон давхарга тодорхойлох
class Dense(flax.nn.Module):
    def apply(self,x,
            features,
            kernel_initialization = jax.nn.initializers.lecun_normal(),
            bias_initialization   = jax.nn.initializers.zeros
            ):
        input_features = x.shape[-1]
        kernel_shape   = (input_features, features)
        kernel         = self.param('kernel', kernel_shape, kernel_initialization)
        bias           = self.param('bias'  , (features,) , bias_initialization  )
        return jnp.dot(x, kernel) + bias


# Неорон сүлжээний оролтын өгөгдлийг бэлтгэх
# энд 1 ширхэг дата буюу batch нь 1 гэсэн үг 4-н оролтын өгөгдлийн дүрс
x = jnp.ones((1, 4))

# Оролтын өгөгдлийн дүрсээс хамааруулан неорон сүлжээний жингүүдийг 
# эхний байдлаар цэнэглэн гаргаж авах
y, params = Dense.init(jax.random.PRNGKey(0), x, features=4)
print("Неорон сүлжээний эхний цэнэглэгдсэн параметрүүд :")
print(params)
print("Эхний байдлаар predict хийсэн гаралт :")
print(y)

# Цэнэглэгдсэн параметрүүдээ хэрэглэн энгийн predict хийх forward дуудалт
prediction = Dense.call(params, x, features=4)
print("predict хийсэн дуудалт :")
print(prediction)


# Зөвхөн hyperparameter-үүд тохируулсан модуль үүсгэе гэвэл partial функц ашиглаж болно
module    = Dense.partial(features=4)
# Дараа нь жингүүдээ цэнэглэж болно
_, params = module.init(jax.random.PRNGKey(0), x)
print("Модуль үүсгэсний дараа цэнэглэсэн жингүүд :")
print(params)
# Модуль хэрэглэн forward дуудалт хийх
prediction = module.call(params, x)
print("Модулиас хийсэн prediction дуудалтын үр дүн :")
print(prediction)


# Модуль ашигласнаар хооронд нь хольж бүрдүүлж болдог нарийн 
# бүтэцтэй неорон сүлжээ зохион байгуулах боломж бүрддэг

def relu(x):
    return jnp.maximum(0., x)

# Multi Layer Perceptron буюу олон давхаргатай хялбар неорон сүлжээ
class MLP(flax.nn.Module):
    def apply(self, x,
            hidden_features,
            output_features,
            activation_fn):
        z = Dense(x, hidden_features)
        h = activation_fn(z)
        y = Dense(h, output_features)
        return y

module    = MLP.partial(hidden_features=8, output_features=4, activation_fn=relu)
y, params = module.init(jax.random.PRNGKey(0), x)
# Нарийн бүтэцтэй олон давхаргатай неорон сүлжээ бий болгосон учраас 
# параметрүүд нь шаталсан бүтэцтэй dictionary байна
print("MLP параметрүүд :")
print(params)
print("MLP predict хийсэн утгууд :")
print(y)


# Хэрвээ неорон давхаргад нэр оноогоогүй байвал автоматаар дугаарлагдаад явна
# тогтсон нэр өгч ашиглах боломжтой

class NamedMLP(flax.nn.Module):
    def apply(self, x,
            hidden_features,
            output_features,
            activation_fn):
        z = Dense(x, hidden_features, name='hidden')
        h = activation_fn(z)
        y = Dense(h, output_features, name='out')
        return y

module    = NamedMLP.partial(hidden_features=8, output_features=4, activation_fn=relu)
_, params = module.init(jax.random.PRNGKey(0), x)

print("Нэрлэсэн давхаргууд бүхий неорон сүлжээний бүтэц :")
print(jax.tree_map(np.shape, params))


# Олон call дуудалтанд дундын parameter-рүүд хэрэглэж болно

class SimpleRNN(flax.nn.Module):
    def apply(self, x, iterations=3):
        dense = Dense.shared(features=x.shape[-1],
                kernel_initialization=jax.nn.initializers.orthogonal(), name='cell')
        ys = []
        for i in range(iterations):
            x = dense(x)
            ys.append(x)
        return ys

ys, params = SimpleRNN.init(jax.random.PRNGKey(0), x)
print("Хялбар Recurrent Neural Network-ийн гаралт :")
print(ys)

# cell давхаргыг 3-н удаа дуудсанч shared параметер горимоор үүсгээн тул нэмэлтээр 
# 3-н давхарга болж орж ирэхгүй нэг л давхарга байдлаар харагдана гэсэн үг
# {'cell': {'bias': (4,), 'kernel': (4, 4)}}
print("Recurrent Neural Network-ийн параметер бүтэц :")
print(jax.tree_map(np.shape, params))


# Оролтын өгөгдлийн дүрснээс хамаарч модулийг init хийх нь заримдаа шаардлагагүй нэмэлт 
# үйлдлүүд хийгдэх магадлалтай, зөвхөн оролтын дүрс нь ямар вэ гэдгээр init хийж болно
# {'cell': {'bias': (2,), 'kernel': (2, 2)}}
input_spec       = [(1,2)]
out_spec, params = SimpleRNN.init_by_shape(jax.random.PRNGKey(0), input_spec)
print("Shape inference горимоор барьсан неорон сүлжээний бүтэц :")
print(jax.tree_map(np.shape, params))


# Модуль хэрэглэснээр неорон сүлжээний параметрүүдийг init болон call функцын тусламжтайгаар
# ил хэлбэрээр track хийх боломж олгодог. Model класс бол Module классын багахан өргөтсөн
# хэлбэр бөгөөд заавал параметрүүдийг нь track хийж байн байн дамжуулаад байх шаардлагагүй
x      = jnp.ones((1,2))
module = Dense.partial(features=4)
ys, initial_params = module.init(jax.random.PRNGKey(0), x)
model  = flax.nn.Model(module, initial_params)
print("Моделийн бүтцийг харах :")
print(jax.tree_map(np.shape, model.params))

prediction = model(x)
print("Моделиор predict хийсэн үр дүн :")
print(prediction)

print("Моделийн параметрүүд :")
print(model.params)


# Моделийн параметрүүдийг нь шинэчилж сольж болно
biased_model = model.replace(
        params={
            'kernel': model.params['kernel'],
            'bias'  : model.params['bias'  ]+1.
            })

print("Параметрүүдийг нь шинэчилсэн моделийн параметрүүд :")
print(biased_model.params)


# Модель нь JAX-ны pytree объект байдлаар зохион байгуулагддаг тул 
# JAX-н native хувиргалтуудад асуудалгүйгээр хэрэглэх боломжтой

# Жишээлбэл модель объект хэрэглэн градиент утгууд олж болно
def loss_fn(model):
    y = model(x)
    return jnp.mean(jnp.square(y))

model_gradients = jax.grad(loss_fn)(model)
print("Моделийн loss-оос авсан gradient утгууд")
print(model_gradients.params)


# BatchNorm мэтийн тохиолдолд batch-уудын mean, variance гэх мэт статистик өгөгдлүүдийг
# хадгалаж хэрэглэх хэрэгцээ гардаг энэ тохиолдолд Module.state-г хэрэглэж болно
class BatchNorm(flax.nn.Module):
    def apply(self, x,
            red_axis   = 0,
            eps        = 1e-5,
            momentum   = 0.99,
            training   = False,
            gamma_init = jax.nn.initializers.ones,
            beta_init  = jax.nn.initializers.zeros
            ):
        # Оролтын моментүүдийг тооцоолох
        mean    = x.mean(red_axis, keepdims=True)
        var     = jnp.square(x-mean).mean(red_axis, keepdims=True)
        # State хувьсагчууд тодорхойлох
        ra_mean = self.state('mean', mean.shape, jax.nn.initializers.zeros)
        ra_var  = self.state('var' , var.shape , jax.nn.initializers.ones )
        # Жингүүдийг эхний байдлаар цэнэглэж байгаа бол 
        # average утгуудыг тооцох шаардлага байхгүй
        if not self.is_initializing():
            if training:
                # Сургаж байгаа үед average-үүдийг шинэчлэх ёстой
                alpha          = 1. - momentum
                ra_mean.value += alpha*(mean - ra_mean.value)
                ra_var.value  += alpha*(var  - ra_var.value )
            else:
                # Сургаагүй тохиолдолд average-уудаа хэрэглэнэ
                mean = ra_mean.value
                var  = ra_var.value
        # Оролтын өгөгдлөө стандартчилах
        y = (x-mean)/jnp.sqrt(var+eps)
        # Оролтын scale болон bias-уудыг суралцах
        gamma = self.param('gamma', mean.shape, gamma_init)
        beta  = self.param('beta' , mean.shape, beta_init )
        return gamma*y+beta

# Хэрэв state хэрэглэхээр бол flax.nn.stateful context manager ашиглана
# Энэ state-үүд flax.nn.Collection дотор хадгалагддаг

class MyModel(flax.nn.Module):
    def apply(self, x, training=False):
        x = Dense(x, features=4)
        x = BatchNorm(x, training=training, momentum=0., name='batch_norm')
        return x

dist_a = lambda rng, shape: jax.random.normal(rng, shape)*jnp.array([[1., 3.]])
x_a    = dist_a(jax.random.PRNGKey(1), (1024, 2))
print("Оролтын өгөгдлын стандарт хазайлт :", x_a.std(0))

with flax.nn.stateful() as init_state:
    y, params = MyModel.init(jax.random.PRNGKey(2), x_a)
print("Гаралтын стандарт хазайлт (init) :", y.std(0))

with flax.nn.stateful(init_state) as new_state:
    y = MyModel.call(params, x_a, training=True)
print("Гаралтын стандарт хазайлт (training) :", y.std(0))

with flax.nn.stateful(new_state, mutable=False):
    y = MyModel.call(params, x_a, training=False)
print("Гаралтын стандарт хазайлт (testing) :", y.std(0))

# state өгөгдлийг Collection.as_dict() ээр шалгаж болно
print("Эхлэл state :")
print(init_state.as_dict())

print("Шинэ state :")
print(new_state.as_dict())

# state тооцох механизм нь ил байх ёстой жишээлбэл 
# тест хийж байх үед state тооцоолох шаардлагагүй
# мөн өөр оролтын өгөгдлүүд хэрэглэн өөр статистик 
# state цуглуулах шаардлага гардаг энэ тохиолд
# ил байдлаар state тооцохын үр ашиг гардаг
dist_b = lambda rng, shape: jax.random.normal(rng, shape)*jnp.array([[2., 5.]])
x_b    = dist_b(jax.random.PRNGKey(1), (1024, 2))
with flax.nn.stateful(new_state, mutable=False):
    y = MyModel.call(params, x_b, training=False)
print(y.std(0)) # Энэ тохиолдолд зөв нормчилогдож чадахгүй

with flax.nn.stateful(init_state) as state_b:
    y = MyModel.call(params, x_b, training=True)
print("Гаралтын стандарт хазайлт (training) :", y.std(0))

with flax.nn.stateful(state_b, mutable=False):
    y = MyModel.call(params, x_b, training=False)
print("Гаралтын стандарт хазайлт (testing) :", y.std(0))
