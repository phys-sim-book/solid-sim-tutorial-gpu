#include "simulator.h"
#include <SFML/Graphics.hpp>
#include "InertialEnergy.h"
#include "MassSpringEnergy.h"
#include "GravityEnergy.h"
#include "BarrierEnergy.h"
#include "FrictionEnergy.h"
#include <muda/muda.h>
#include <muda/container.h>
#include "uti.h"
using namespace muda;
template <typename T, int dim>
struct FrictionSimulator<T, dim>::Impl
{
    int n_seg;
    T h, rho, side_len, initial_stretch, m, tol, mu;
    int resolution = 900, scale = 200, offset = resolution / 2, radius = 5;
    std::vector<T> host_x, host_v, host_k, host_l2;
    std::vector<int> host_e;
    DeviceBuffer<T> device_x, device_v;
    DeviceBuffer<int> device_DBC;
    DeviceBuffer<T> device_contact_area;
    sf::RenderWindow window;
    InertialEnergy<T, dim> inertialenergy;
    MassSpringEnergy<T, dim> massspringenergy;
    GravityEnergy<T, dim> gravityenergy;
    BarrierEnergy<T, dim> barrierenergy;
    FrictionEnergy<T, dim> frictionenergy;
    Impl(T rho, T side_len, T initial_stretch, T K, T h_, T tol_, T mu_, int n_seg);
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
FrictionSimulator<T, dim>::FrictionSimulator() = default;

template <typename T, int dim>
FrictionSimulator<T, dim>::~FrictionSimulator() = default;

template <typename T, int dim>
FrictionSimulator<T, dim>::FrictionSimulator(FrictionSimulator<T, dim> &&rhs) = default;

template <typename T, int dim>
FrictionSimulator<T, dim> &FrictionSimulator<T, dim>::operator=(FrictionSimulator<T, dim> &&rhs) = default;

template <typename T, int dim>
FrictionSimulator<T, dim>::FrictionSimulator(T rho, T side_len, T initial_stretch, T K, T h_, T tol_, T mu_, int n_seg) : pimpl_{std::make_unique<Impl>(rho, side_len, initial_stretch, K, h_, tol_, mu_, n_seg)}
{
}
template <typename T, int dim>
FrictionSimulator<T, dim>::Impl::Impl(T rho, T side_len, T initial_stretch, T K, T h_, T tol_, T mu_, int n_seg) : tol(tol_), h(h_), mu(mu_), window(sf::VideoMode(resolution, resolution), "FrictionSimulator")
{
    generate(side_len, n_seg, host_x, host_e);
    std::vector<int> DBC(host_x.size() / dim, 0);
    std::vector<T> contact_area(host_x.size() / dim, side_len / n_seg);
    std::vector<T> ground_n(dim);
    ground_n[0] = 0.1, ground_n[1] = 1;
    T n_norm = ground_n[0] * ground_n[0] + ground_n[1] * ground_n[1];
    n_norm = sqrt(n_norm);
    for (int i = 0; i < dim; i++)
        ground_n[i] /= n_norm;
    std::vector<T> ground_o(dim);
    ground_o[0] = 0, ground_o[1] = -1;
    host_v.resize(host_x.size(), 0);
    host_k.resize(host_e.size() / 2, K);
    host_l2.resize(host_e.size() / 2);
    device_x.copy_from(host_x);
    device_v.copy_from(host_v);
    for (int i = 0; i < host_e.size() / 2; i++)
    {
        T diff = 0;
        int idx1 = host_e[2 * i], idx2 = host_e[2 * i + 1];
        for (int d = 0; d < dim; d++)
        {
            diff += (host_x[idx1 * dim + d] - host_x[idx2 * dim + d]) * (host_x[idx1 * dim + d] - host_x[idx2 * dim + d]);
        }
        host_l2[i] = diff;
    }
    m = rho * side_len * side_len / ((n_seg + 1) * (n_seg + 1));
    // initial stretch
    int N = host_x.size() / dim;
    for (int i = 0; i < N; i++)
        host_x[i * dim + 0] *= initial_stretch;
    inertialenergy = InertialEnergy<T, dim>(N, m);
    massspringenergy = MassSpringEnergy<T, dim>(host_x, host_e, host_l2, host_k);
    gravityenergy = GravityEnergy<T, dim>(N, m);
    barrierenergy = BarrierEnergy<T, dim>(host_x, ground_n, ground_o, contact_area);
    frictionenergy = FrictionEnergy<T, dim>(host_v, h, ground_n);
    device_DBC = DeviceBuffer<int>(DBC);
    device_contact_area = DeviceBuffer<T>(contact_area);
}
template <typename T, int dim>
void FrictionSimulator<T, dim>::run()
{
    assert(dim == 2);
    bool running = true;
    auto &window = pimpl_->window;
    int time_step = 0;
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
        std::cout << "Time step " << time_step++ << "\n";
        pimpl_->step_forward();
    }

    window.close();
}

template <typename T, int dim>
void FrictionSimulator<T, dim>::Impl::step_forward()
{
    update_x_tilde(add_vector<T>(device_x, device_v, 1, h));
    barrierenergy.compute_mu_lambda(mu, frictionenergy.get_mu_lambda());
    DeviceBuffer<T> device_x_n = device_x; // Copy current positions to device_x_n
    update_v(add_vector<T>(device_x, device_x_n, 1 / h, -1 / h));
    int iter = 0;
    T E_last = IP_val();
    DeviceBuffer<T> device_p = search_direction();
    T residual = max_vector(device_p) / h;
    // std::cout << "Initial residual " << residual << "\n";
    while (residual > tol)
    {
        std::cout << "Iteration " << iter << " residual " << residual << " E_last" << E_last << "\n";
        // Line search
        T alpha = barrierenergy.init_step_size(device_p);
        DeviceBuffer<T> device_x0 = device_x;
        update_x(add_vector<T>(device_x0, device_p, 1.0, alpha));
        update_v(add_vector<T>(device_x, device_x_n, 1 / h, -1 / h));
        while (IP_val() > E_last)
        {
            alpha /= 2;
            update_x(add_vector<T>(device_x0, device_p, 1.0, alpha));
            update_v(add_vector<T>(device_x, device_x_n, 1 / h, -1 / h));
        }
        std::cout << "step size = " << alpha << "\n";
        E_last = IP_val();
        device_p = search_direction();
        residual = max_vector(device_p) / h;
        iter += 1;
    }
    update_v(add_vector<T>(device_x, device_x_n, 1 / h, -1 / h));
}
template <typename T, int dim>
T FrictionSimulator<T, dim>::Impl::screen_projection_x(T point)
{
    return offset + scale * point;
}
template <typename T, int dim>
T FrictionSimulator<T, dim>::Impl::screen_projection_y(T point)
{
    return resolution - (offset + scale * point);
}
template <typename T, int dim>
void FrictionSimulator<T, dim>::Impl::update_x(const DeviceBuffer<T> &new_x)
{
    inertialenergy.update_x(new_x);
    massspringenergy.update_x(new_x);
    gravityenergy.update_x(new_x);
    barrierenergy.update_x(new_x);
    device_x = new_x;
}
template <typename T, int dim>
void FrictionSimulator<T, dim>::Impl::update_x_tilde(const DeviceBuffer<T> &new_x_tilde)
{
    inertialenergy.update_x_tilde(new_x_tilde);
}
template <typename T, int dim>
void FrictionSimulator<T, dim>::Impl::update_v(const DeviceBuffer<T> &new_v)
{
    frictionenergy.update_v(new_v);
    device_v = new_v;
}
template <typename T, int dim>
void FrictionSimulator<T, dim>::Impl::draw()
{
    device_x.copy_to(host_x);
    window.clear(sf::Color::White); // Clear the previous frame

    // Draw the ground
    sf::Vertex line[] = {
        sf::Vertex(sf::Vector2f(screen_projection_x(0 - 1 * 5), screen_projection_y(-1.0 + 0.1 * 5)), sf::Color::Blue),
        sf::Vertex(sf::Vector2f(screen_projection_x(0 + 1 * 5), screen_projection_y(-1.0 - 0.1 * 5)), sf::Color::Blue)};
    window.draw(line, 2, sf::Lines);

    // Draw springs as lines
    for (int i = 0; i < host_e.size() / 2; ++i)
    {
        sf::Vertex line[] = {
            sf::Vertex(sf::Vector2f(screen_projection_x(host_x[host_e[i * 2] * dim]), screen_projection_y(host_x[host_e[i * 2] * dim + 1])), sf::Color::Blue),
            sf::Vertex(sf::Vector2f(screen_projection_x(host_x[host_e[i * 2 + 1] * dim]), screen_projection_y(host_x[host_e[i * 2 + 1] * dim + 1])), sf::Color::Blue)};
        window.draw(line, 2, sf::Lines);
    }

    // Draw masses as circles
    for (int i = 0; i < host_x.size() / dim; ++i)
    {
        sf::CircleShape circle(radius); // Set a fixed radius for each mass
        circle.setFillColor(sf::Color::Red);
        circle.setPosition(screen_projection_x(host_x[i * dim]) - radius, screen_projection_y(host_x[i * dim + 1]) - radius); // Center the circle on the mass
        window.draw(circle);
    }

    window.display(); // Display the rendered frame
}

template <typename T, int dim>
T FrictionSimulator<T, dim>::Impl::IP_val()
{

    return inertialenergy.val() + (massspringenergy.val() + gravityenergy.val() + barrierenergy.val() + frictionenergy.val()) * h * h;
}

template <typename T, int dim>
DeviceBuffer<T> FrictionSimulator<T, dim>::Impl::IP_grad()
{
    return add_vector<T>(add_vector<T>(add_vector<T>(add_vector<T>(inertialenergy.grad(), massspringenergy.grad(), 1.0, h * h), gravityenergy.grad(), 1.0, h * h), barrierenergy.grad(), 1.0, h * h), frictionenergy.grad(), 1.0, h * h);
}

template <typename T, int dim>
DeviceTripletMatrix<T, 1> FrictionSimulator<T, dim>::Impl::IP_hess()
{
    DeviceTripletMatrix<T, 1> inertial_hess = inertialenergy.hess();
    DeviceTripletMatrix<T, 1> massspring_hess = massspringenergy.hess();
    DeviceTripletMatrix<T, 1> hess = add_triplet<T>(inertial_hess, massspring_hess, 1.0, h * h);
    DeviceTripletMatrix<T, 1> barrier_hess = barrierenergy.hess();
    hess = add_triplet<T>(hess, barrier_hess, 1.0, h * h);
    DeviceTripletMatrix<T, 1> friction_hess = frictionenergy.hess();
    hess = add_triplet<T>(hess, friction_hess, 1.0, h * h);
    return hess;
}
template <typename T, int dim>
DeviceBuffer<T> FrictionSimulator<T, dim>::Impl::search_direction()
{
    DeviceBuffer<T> dir;
    dir.resize(host_x.size());
    DeviceBuffer<T> grad = IP_grad();
    DeviceTripletMatrix<T, 1> hess = IP_hess();
    search_dir<T, dim>(grad, hess, dir, device_DBC);
    return dir;
}

template class FrictionSimulator<float, 2>;
template class FrictionSimulator<double, 2>;
template class FrictionSimulator<float, 3>;
template class FrictionSimulator<double, 3>;