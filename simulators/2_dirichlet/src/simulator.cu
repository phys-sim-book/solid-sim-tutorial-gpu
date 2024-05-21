#include "simulator.h"
#include <SFML/Graphics.hpp>
#include "InertialEnergy.h"
#include "MassSpringEnergy.h"
#include "GravityEnergy.h"
#include <muda/muda.h>
#include <muda/container.h>
#include "uti.h"
using namespace muda;
template <typename T, int dim>
struct DirichletSimulator<T, dim>::Impl
{
    int n_seg;
    T h, rho, side_len, initial_stretch, m, tol;
    int resolution = 900, scale = 200, offset = resolution / 2, radius = 5;
    std::vector<T> x, x_tilde, v, k, l2;
    std::vector<int> e;
    DeviceBuffer<int> device_DBC;
    sf::RenderWindow window;
    InertialEnergy<T, dim> inertialenergy;
    MassSpringEnergy<T, dim> massspringenergy;
    GravityEnergy<T, dim> gravityenergy;
    Impl(T rho, T side_len, T initial_stretch, T K, T h_, T tol_, int n_seg);
    void update_x(const DeviceBuffer<T> &new_x);
    void update_x_tilde(const DeviceBuffer<T> &new_x_tilde);
    void update_v(const DeviceBuffer<T> &new_v);
    T IP_val();
    void step_forward();
    void draw();
    DeviceBuffer<T> IP_grad();
    DeviceTripletMatrix<T, 1> IP_hess();
    DeviceBuffer<T> search_direction();
    T screen_projection_x(T point);
    T screen_projection_y(T point);
};
template <typename T, int dim>
DirichletSimulator<T, dim>::DirichletSimulator() = default;

template <typename T, int dim>
DirichletSimulator<T, dim>::~DirichletSimulator() = default;

template <typename T, int dim>
DirichletSimulator<T, dim>::DirichletSimulator(DirichletSimulator<T, dim> &&rhs) = default;

template <typename T, int dim>
DirichletSimulator<T, dim> &DirichletSimulator<T, dim>::operator=(DirichletSimulator<T, dim> &&rhs) = default;

template <typename T, int dim>
DirichletSimulator<T, dim>::DirichletSimulator(T rho, T side_len, T initial_stretch, T K, T h_, T tol_, int n_seg) : pimpl_{std::make_unique<Impl>(rho, side_len, initial_stretch, K, h_, tol_, n_seg)}
{
}
template <typename T, int dim>
DirichletSimulator<T, dim>::Impl::Impl(T rho, T side_len, T initial_stretch, T K, T h_, T tol_, int n_seg) : tol(tol_), h(h_), window(sf::VideoMode(resolution, resolution), "DirichletSimulator")
{
    generate(side_len, n_seg, x, e);
    std::vector<int> DBC(x.size() / dim, 0);
    DBC[n_seg] = 1;
    DBC[(n_seg + 1) * (n_seg + 1) - 1] = 1;
    v.resize(x.size(), 0);
    k.resize(e.size() / 2, K);
    l2.resize(e.size() / 2);
    for (int i = 0; i < e.size() / 2; i++)
    {
        T diff = 0;
        int idx1 = e[2 * i], idx2 = e[2 * i + 1];
        for (int d = 0; d < dim; d++)
        {
            diff += (x[idx1 * dim + d] - x[idx2 * dim + d]) * (x[idx1 * dim + d] - x[idx2 * dim + d]);
        }
        l2[i] = diff;
    }
    m = rho * side_len * side_len / ((n_seg + 1) * (n_seg + 1));
    // initial stretch
    int N = x.size() / dim;
    for (int i = 0; i < N; i++)
        x[i * dim + 0] *= initial_stretch;
    inertialenergy = InertialEnergy<T, dim>(N, m);
    massspringenergy = MassSpringEnergy<T, dim>(x, e, l2, k);
    gravityenergy = GravityEnergy<T, dim>(N, m);
    DeviceBuffer<T> x_device(x);
    update_x(x_device);
    device_DBC = DeviceBuffer<int>(DBC);
}
template <typename T, int dim>
void DirichletSimulator<T, dim>::run()
{
    assert(dim == 2);
    bool running = true;
    auto &window = pimpl_->window;
    while (running)
    {
        sf::Event event;
        while (window.pollEvent(event))
        {
            if (event.type == sf::Event::Closed)
                running = false;
        }

        pimpl_->draw(); // Draw the current state

        // Update the simulation state
        pimpl_->step_forward();
    }

    window.close();
}

template <typename T, int dim>
void DirichletSimulator<T, dim>::Impl::step_forward()
{
    DeviceBuffer<T> x_tilde(x.size()); // Predictive position
    update_x_tilde(add_vector<T>(x, v, 1, h));
    DeviceBuffer<T> x_n = x; // Copy current positions to x_n
    int iter = 0;
    T E_last = IP_val();
    DeviceBuffer<T> p = search_direction();
    T residual = max_vector(p) / h;
    // std::cout << "Initial residual " << residual << "\n";
    while (residual > tol)
    {
        // Line search
        T alpha = 1;
        DeviceBuffer<T> x0 = x;
        update_x(add_vector<T>(x0, p, 1.0, alpha));
        while (IP_val() > E_last)
        {
            alpha /= 2;
            update_x(add_vector<T>(x0, p, 1.0, alpha));
        }
        // std::cout << "step size = " << alpha << "\n";
        E_last = IP_val();
        // std::cout << "Iteration " << iter << " residual " << residual << "E_last" << E_last << "\n";
        p = search_direction();
        residual = max_vector(p) / h;
        iter += 1;
    }
    update_v(add_vector<T>(x, x_n, 1 / h, -1 / h));
}
template <typename T, int dim>
T DirichletSimulator<T, dim>::Impl::screen_projection_x(T point)
{
    return offset + scale * point;
}
template <typename T, int dim>
T DirichletSimulator<T, dim>::Impl::screen_projection_y(T point)
{
    return resolution - (offset + scale * point);
}
template <typename T, int dim>
void DirichletSimulator<T, dim>::Impl::update_x(const DeviceBuffer<T> &new_x)
{
    inertialenergy.update_x(new_x);
    massspringenergy.update_x(new_x);
    gravityenergy.update_x(new_x);
    new_x.copy_to(x);
}
template <typename T, int dim>
void DirichletSimulator<T, dim>::Impl::update_x_tilde(const DeviceBuffer<T> &new_x_tilde)
{
    inertialenergy.update_x_tilde(new_x_tilde);
    new_x_tilde.copy_to(x_tilde);
}
template <typename T, int dim>
void DirichletSimulator<T, dim>::Impl::update_v(const DeviceBuffer<T> &new_v)
{
    new_v.copy_to(v);
}
template <typename T, int dim>
void DirichletSimulator<T, dim>::Impl::draw()
{
    window.clear(sf::Color::White); // Clear the previous frame

    // Draw springs as lines
    for (int i = 0; i < e.size() / 2; ++i)
    {
        sf::Vertex line[] = {
            sf::Vertex(sf::Vector2f(screen_projection_x(x[e[i * 2] * dim]), screen_projection_y(x[e[i * 2] * dim + 1])), sf::Color::Blue),
            sf::Vertex(sf::Vector2f(screen_projection_x(x[e[i * 2 + 1] * dim]), screen_projection_y(x[e[i * 2 + 1] * dim + 1])), sf::Color::Blue)};
        window.draw(line, 2, sf::Lines);
    }

    // Draw masses as circles
    for (int i = 0; i < x.size() / dim; ++i)
    {
        sf::CircleShape circle(radius); // Set a fixed radius for each mass
        circle.setFillColor(sf::Color::Red);
        circle.setPosition(screen_projection_x(x[i * dim]) - radius, screen_projection_y(x[i * dim + 1]) - radius); // Center the circle on the mass
        window.draw(circle);
    }

    window.display(); // Display the rendered frame
}

template <typename T, int dim>
T DirichletSimulator<T, dim>::Impl::IP_val()
{

    return inertialenergy.val() + massspringenergy.val() * h * h + gravityenergy.val() * h * h;
}

template <typename T, int dim>
DeviceBuffer<T> DirichletSimulator<T, dim>::Impl::IP_grad()
{
    return add_vector<T>(add_vector<T>(inertialenergy.grad(), massspringenergy.grad(), 1.0, h * h), gravityenergy.grad(), 1.0, h * h);
}

template <typename T, int dim>
DeviceTripletMatrix<T, 1> DirichletSimulator<T, dim>::Impl::IP_hess()
{
    DeviceTripletMatrix<T, 1> inertial_hess = inertialenergy.hess();
    DeviceTripletMatrix<T, 1> massspring_hess = massspringenergy.hess();
    DeviceTripletMatrix<T, 1> hess = add_triplet<T>(inertial_hess, massspring_hess, 1.0, h * h);
    return hess;
}
template <typename T, int dim>
DeviceBuffer<T> DirichletSimulator<T, dim>::Impl::search_direction()
{
    DeviceBuffer<T> dir;
    dir.resize(x.size());
    DeviceBuffer<T> grad = IP_grad();
    DeviceTripletMatrix<T, 1> hess = IP_hess();
    search_dir<T, dim>(grad, hess, dir, device_DBC);
    return dir;
}

template class DirichletSimulator<float, 2>;
template class DirichletSimulator<double, 2>;
template class DirichletSimulator<float, 3>;
template class DirichletSimulator<double, 3>;