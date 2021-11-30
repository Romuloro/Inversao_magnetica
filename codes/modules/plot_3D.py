import numpy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection


def plot_prism(vert, color):
    fig01 = Poly3DCollection(vert, alpha = 0.75, linewidths = 0.75, edgecolors = 'k')
    fig01.set_facecolor(color)
    return fig01

def plot_obs_3d(prism, size, view, x, y, z):
    
    '''
    prism = dicionário com o número de prismas que seram plotados e os prismas desenhados.
        prism = {'n': n, - números de prismas que seram plotados
                 'prisma': [prisma1, prisma2, ..., prismaN] - todos os prismas que serão incluidos}
    size = tamanho da figura
    view = ângulo de visualização
    x = matriz de posições em x
    y = matrix de posições em y
    z = matriz profundidades do prisma
    '''
    figure = plt.figure(figsize=(size[0],size[1]))
    ax = figure.gca(projection = '3d')
    #----------------------------------------------------------------------------------------------------#
    n = prism.get('n')
    prisma = prism.get('prisma')
    for i in range(n):
        ax.add_collection3d(prisma[i])
    #----------------------------------------------------------------------------------------------------#
    # Define the scale of the projection
    x_scale = 1.2
    y_scale = 1.2
    z_scale = 1.
    scale=numpy.diag([x_scale, y_scale, z_scale, 1.0])
    scale=scale*(1.0/scale.max())
    scale[3,3] = 1.

    # Labels
    ax.set_xlabel('East (km)', size = 25, labelpad = 30)
    ax.set_ylabel('North (km)', size = 25, labelpad = 30)
    ax.set_zlabel('Depth (km)', size = 25, labelpad = 30)
    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(y.min(), y.max())
    ax.set_zlim(min(z)-500.0, max(z)+500.0)
    #ax.set_xticks(numpy.arange(x.min(), x.max(), 2500))
    #ax.set_yticks(numpy.linspace(y.min(), y.max(), 5))
    #ax.set_zticks(numpy.linspace(0., z[1], 6))
    #ax.tick_params(labelsize = 20, pad = 10)

    # Visualization angle
    ax.view_init(view[0], view[1])

    plt.tight_layout(True)
    #plt.savefig('prisma_3D.pdf', format='pdf')

    plt.show()
    
def creat_point (n, x, y, z, deltay, deltaz, deltax, incl, merg):
    '''
    As entradas da função é feita da forma clássica ou através de um dicionário que é descompactado.
    O dicinário deve conter as chaves nomeadas de forma identica aos parâmetros de entrada da função.
    Exemplo de entrada: creat_point(**dicionario).

    dicionario = dicionário com todos as entradas organizadas da seguinte forma.
    dicionario = {'n': n, - número de interações
                  'x': [x1, x2], - primeiras coordenadas de x
                  'y': [y1, y2], - primeiras coordenadas de y
                  'z': [z1, z2], - primeiras coordenadas de z
                  'deltay': 'deltay', - valor de deltay
                  'deltaz': 'deltaz', - valor de deltax
                  'incl': 'positivo' or 'negativo', - direção de mergulho da escada. Positivo(esquerda -> direta); Negativo(direta -> esquerda)}
    '''

    pointx = []
    pointy = []
    pointz = []
    
    if merg == 'y':
        
    # Criação dos pontos em X
        for i in range(n):
            pointx.extend([x[0],x[1]])

    # Criação dos pontos em Z
        for i in range(n):
            pointz.extend([z[0]+i*deltaz/2,z[1]+i*deltaz/2])
    
    # Criação dos pontos em Y
        if incl == 'positivo':
            for i in range(n):
                pointy.extend([y[0]+i*deltay/2,y[1]+i*deltay/2])
        else:
            for i in range(n):
                pointy.extend([y[0]-i*deltay/2,y[1]-i*deltay/2])
    
    else:
        # Criação dos pontos em Y
        for i in range(n):
            pointy.extend([y[0],y[1]])

    # Criação dos pontos em Z
        for i in range(n):
            pointz.extend([z[0]+i*deltaz/2,z[1]+i*deltaz/2])
    
    # Criação dos pontos em z
        if incl == 'positivo':
            for i in range(n):
                pointx.extend([x[0]+i*deltax/2,x[1]+i*deltax/2])
        else:
            for i in range(n):
                pointx.extend([x[0]-i*deltax/2,x[1]-i*deltax/2])
    
    return pointx, pointy, pointz

def vert_point (dicionario):
    '''
    Função com o objetivo de criar uma array com os verteces, para a construção de um prisma.
    
    dicionario = dicionário com todos as entradas organizadas da seguinte forma.
    dicionario = {'x': [x1, x2], - coordenadas de x
                  'y': [y1, y2], - coordenadas de y
                  'z': [z1, z2], - coordenadas de z}
    '''
    x = dicionario.get('x')
    y = dicionario.get('y')
    z = dicionario.get('z')
    
    v = numpy.array([[x[0], y[0], z[1]], [x[0], y[1], z[1]], [x[1], y[1], z[1]], [x[1], y[0], z[1]], 
                 [x[0], y[0], z[0]], [x[0], y[1], z[0]], [x[1], y[1], z[0]], [x[1], y[0], z[0]]])
    #----------------------------------------------------------------------------------------------------#
    vert =  [[v[0],v[1],v[2],v[3]], 
            [v[0],v[1],v[5],v[4]], 
            [v[1],v[2],v[6],v[5]],
            [v[2],v[3],v[7],v[6]], 
            [v[3],v[0],v[4],v[7]], 
            [v[4],v[5],v[6],v[7]]]
    
    return vert

def create_aquisicao(nx, ny, xmin, xmax, ymin, ymax, z, color):
    '''
    As entradas da função é feita da forma clássica ou através de um dicionário que é descompactado.
    O dicinário deve conter as chaves nomeadas de forma identica aos parâmetros de entrada da função.
    Exemplo de entrada: create_aquisicao(**dicionario).

    dicionario = dicionário com todos as entradas organizadas da seguinte forma.
    dicionario = {'nx': [nx], - número de observações em x
                  'ny': [ny], - número de observações em y
                  'xmin': [x1], - Limite das coordenadas positivas de x
                  'xmax': [x2], - Limite das coordenadas negativas de x
                  'ymin': [y1], - Limite das coordenadas positivas de y
                  'ymax': [y2], - Limite das coordenadas negativas de y
                  'z': [z1], - altura de voo, em metros
                  'color': 'r', a cor escolhida para o plot da malha}
    '''

    x = numpy.linspace(xmin, xmax, nx, endpoint=True)
    y = numpy.linspace(ymin, ymax, ny, endpoint=True)
    #----------------------------------------------------------------------------------------------------#
    X,Y = numpy.meshgrid(x,y)
    Z = numpy.copy(X)*0.0 + z
    #----------------------------------------------------------------------------------------------------#
    plt.figure(figsize=(12,12))
    plt.title('Levantamento aéreo')
    plt.plot(X,Y, '.r')
    plt.show()
    
    return x, y, X, Y, Z

def modelo_anomalia_3D(Yref, Xref, tfa_n_bolinhas, resulty, resultx, resultz, resultadoz):
    """
    Função com o objetivo de plotar todo o modelo 3D de bolinhas juntamente com a anomalia magnética
    criada por elas.

    As entradas da função é feita da forma clássica ou através de um dicionário que é descompactado.
    O dicinário deve conter as chaves nomeadas de forma identica aos parâmetros de entrada da função.
    Exemplo de entrada: modelo_anomalia_3D(**dicionario).

    """

    view = [190, 130]
    fig = plt.figure(figsize=(15, 15))
    ax = fig.gca(projection='3d')
    plano = ax.contourf(Yref, Xref, tfa_n_bolinhas, offset=0, cmap=plt.cm.RdBu_r)
    p = ax.scatter(resulty, resultx, resultz, c=resultadoz, depthshade=True, cmap='rainbow')
    #plt.xlim(-5000, 5000)
    #plt.ylim(-5000, 5000)
    ax.set_xlabel('East (m)', fontsize=20)
    ax.set_ylabel('North (m)', fontsize=20)
    ax.set_zlabel('Depth (m)', fontsize=20)
    plt.colorbar(plano, shrink=0.65)
    plt.grid()
    ax.view_init(view[0], view[1])
    plt.show()


def plot_obs_3d_bolinha(prism, size, view, x, y, z, coodX1, coodY1, coodZ1, incl1):
    
    '''
    prism = dicionário com o número de prismas que seram plotados e os prismas desenhados.
        prism = {'n': n, - números de prismas que seram plotados
                 'prisma': [prisma1, prisma2, ..., prismaN] - todos os prismas que serão incluidos}
    size = tamanho da figura
    view = ângulo de visualização
    x = matriz de posições em x
    y = matrix de posições em y
    z = matriz profundidades do prisma
    '''
    figure = plt.figure(figsize=(size[0],size[1]))
    ax1 = figure.gca(projection = '3d')
    #----------------------------------------------------------------------------------------------------#
    n = prism.get('n')
    prisma = prism.get('prisma')
    for i in range(n):
        ax1.add_collection3d(prisma[i])
    #----------------------------------------------------------------------------------------------------#
    individuo0 = ax.scatter(coodX1, coodY1, coodZ1, c=incl1, depthshade=True, cmap='jet', s = 200.0)
    # Define the scale of the projection
    x_scale = 1.2
    y_scale = 1.2
    z_scale = 1.
    scale=numpy.diag([x_scale, y_scale, z_scale, 1.0])
    scale=scale*(1.0/scale.max())
    scale[3,3] = 1.

    # Labels
    ax.set_xlabel('North (km)', size = 25, labelpad = 30)
    ax.set_ylabel('East (km)', size = 25, labelpad = 30)
    ax.set_zlabel('Depth (km)', size = 25, labelpad = 30)
    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(y.min(), y.max())
    ax.set_zlim(min(z)-500.0, max(z)+500.0)
    #ax.set_xticks(numpy.arange(x.min(), x.max(), 2500))
    #ax.set_yticks(numpy.linspace(y.min(), y.max(), 5))
    #ax.set_zticks(numpy.linspace(0., z[1], 6))
    #ax.tick_params(labelsize = 20, pad = 10)

    # Visualization angle
    ax.view_init(view[0], view[1])

    plt.tight_layout(True)
    #plt.savefig('prisma_3D.pdf', format='pdf')

    plt.show()
    