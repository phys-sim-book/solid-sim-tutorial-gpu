#include "simulator.h"
#include <SFML/Graphics.hpp>
#include "InertialEnergy.h"
#include "MassSpringEnergy.h"
#include <muda/muda.h>
#include <muda/container.h>
#include "uti.h"
using namespace muda;
template <typename T, int dim>
struct MassSpringSimulator<T, dim>::Impl
{
    int n_seg;
    T h, rho, side_len, initial_stretch, m, tol;
    int resolution = 900, scale = 200, offset = resolution / 2, radius = 5;
    std::vector<T> host_x, host_k, host_l2;
    std::vector<int> host_e;
    DeviceBuffer<T> device_x, device_v;
    sf::RenderWindow window;
    InertialEnergy<T, dim> inertialenergy;
    MassSpringEnergy<T, dim> massspringenergy;
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
MassSpringSimulator<T, dim>::MassSpringSimulator() = default;

template <typename T, int dim>
MassSpringSimulator<T, dim>::~MassSpringSimulator() = default;

template <typename T, int dim>
MassSpringSimulator<T, dim>::MassSpringSimulator(MassSpringSimulator<T, dim> &&rhs) = default;

template <typename T, int dim>
MassSpringSimulator<T, dim> &MassSpringSimulator<T, dim>::operator=(MassSpringSimulator<T, dim> &&rhs) = default;

template <typename T, int dim>
MassSpringSimulator<T, dim>::MassSpringSimulator(T rho, T side_len, T initial_stretch, T K, T h_, T tol_, int n_seg) : pimpl_{std::make_unique<Impl>(rho, side_len, initial_stretch, K, h_, tol_, n_seg)}
{
}
template <typename T, int dim>
MassSpringSimulator<T, dim>::Impl::Impl(T rho, T side_len, T initial_stretch, T K, T h_, T tol_, int n_seg) : tol(tol_), h(h_), window(sf::VideoMode(resolution, resolution), "MassSpringSimulator")
{
    generate(side_len, n_seg, host_x, host_e);
    host_k.resize(host_e.size() / 2, K);
    host_l2.resize(host_e.size() / 2);
    device_x.resize(host_x.size(), 0);
    device_v.resize(host_x.size(), 0);
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
    device_x.copy_from(host_x);
}
template <typename T, int dim>
void MassSpringSimulator<T, dim>::run()
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
        std::cout << "Time step " << time_step++ << "\n";
        // Update the simulation state
        pimpl_->step_forward();

        // Wait according to time step
        // sf::sleep(sf::milliseconds(static_cast<int>(h * 1000)));
    }

    window.close();
}

// ANCHOR: step_forward
template <typename T, int dim>
void MassSpringSimulator<T, dim>::Impl::step_forward()
{
    update_x_tilde(add_vector<T>(device_x, device_v, 1, h));
    DeviceBuffer<T> device_x_n = device_x; // Copy current positions to device_x_n
    int iter = 0;
    T E_last = IP_val();
    DeviceBuffer<T> device_p = search_direction();
    T residual = max_vector(device_p) / h;
    while (residual > tol)
    {
        std::cout << "Iteration " << iter << " residual " << residual << "E_last" << E_last << "\n";
        // Line search
        T alpha = 1;
        DeviceBuffer<T> device_x0 = device_x;
        update_x(add_vector<T>(device_x0, device_p, 1.0, alpha));
        while (IP_val() > E_last)
        {
            alpha /= 2;
            update_x(add_vector<T>(device_x0, device_p, 1.0, alpha));
        }
        std::cout << "step size = " << alpha << "\n";
        E_last = IP_val();
        device_p = search_direction();
        residual = max_vector(device_p) / h;
        iter += 1;
    }
    update_v(add_vector<T>(device_x, device_x_n, 1 / h, -1 / h));
}
// ANCHOR_END: step_forward

template <typename T, int dim>
T MassSpringSimulator<T, dim>::Impl::screen_projection_x(T point)
{
    return offset + scale * point;
}
template <typename T, int dim>
T MassSpringSimulator<T, dim>::Impl::screen_projection_y(T point)
{
    return resolution - (offset + scale * point);
}

// ANCHOR: update_x
template <typename T, int dim>
void MassSpringSimulator<T, dim>::Impl::update_x(const DeviceBuffer<T> &new_x)
{
    inertialenergy.update_x(new_x);
    massspringenergy.update_x(new_x);
    device_x = new_x;
}
// ANCHOR_END: update_x

template <typename T, int dim>
void MassSpringSimulator<T, dim>::Impl::update_x_tilde(const DeviceBuffer<T> &new_x_tilde)
{
    inertialenergy.update_x_tilde(new_x_tilde);
}
template <typename T, int dim>
void MassSpringSimulator<T, dim>::Impl::update_v(const DeviceBuffer<T> &new_v)
{
    device_v = new_v;
}
template <typename T, int dim>
void MassSpringSimulator<T, dim>::Impl::draw()
{
    device_x.copy_to(host_x);
    window.clear(sf::Color::White); // Clear the previous frame

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

// ANCHOR: IP_val
template <typename T, int dim>
T MassSpringSimulator<T, dim>::Impl::IP_val()
{

    return inertialenergy.val() + massspringenergy.val() * h * h;
}
// ANCHOR_END: IP_val

// ANCHOR: IP_grad and IP_hess
template <typename T, int dim>
DeviceBuffer<T> MassSpringSimulator<T, dim>::Impl::IP_grad()
{
    return add_vector<T>(inertialenergy.grad(), massspringenergy.grad(), 1.0, h * h);
}

template <typename T, int dim>
DeviceTripletMatrix<T, 1> MassSpringSimulator<T, dim>::Impl::IP_hess()
{
    DeviceTripletMatrix<T, 1> inertial_hess = inertialenergy.hess();
    DeviceTripletMatrix<T, 1> massspring_hess = massspringenergy.hess();
    DeviceTripletMatrix<T, 1> hess = add_triplet<T>(inertial_hess, massspring_hess, 1.0, h * h);
    return hess;
}
// ANCHOR_END: IP_grad and IP_hess
template <typename T, int dim>
DeviceBuffer<T> MassSpringSimulator<T, dim>::Impl::search_direction()
{
    DeviceBuffer<T> dir;
    dir.resize(host_x.size());
    search_dir(IP_grad(), IP_hess(), dir);
    return dir;
}

template class MassSpringSimulator<float, 2>;
template class MassSpringSimulator<double, 2>;
template class MassSpringSimulator<float, 3>;
template class MassSpringSimulator<double, 3>;