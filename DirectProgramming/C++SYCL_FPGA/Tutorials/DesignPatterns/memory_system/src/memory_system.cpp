#include "tp3_video.h"
#include "unrolled_loop.hpp"

using namespace std;
using namespace sycl::ext::intel::experimental;

SYCL_EXTERNAL unsigned int traitement_5x5(unsigned int entree[5][5]);

// Forward declare the kernel and pipe names
// (this prevents unwanted name mangling in the optimization report)

// Pipe properties
using PipePropertiesT = decltype(sycl::ext::oneapi::experimental::properties(
    sycl::ext::intel::experimental::bits_per_symbol<8>,
    sycl::ext::intel::experimental::uses_valid<true>,
    sycl::ext::intel::experimental::ready_latency<0>,
    sycl::ext::intel::experimental::first_symbol_in_high_order_bits<true>));

// Image streams
class flux_in_id;
using flux_in = pipe<
    flux_in_id,     // An identifier for the pipe
    unsigned int,   // The type of data in the pipe
    0,              // The capacity of the pipe
    PipePropertiesT // Customizable pipe properties
    >;

class flux_out_id;
using flux_out = pipe<
    flux_out_id,    // An identifier for the pipe
    unsigned int,   // The type of data in the pipe
    0,              // The capacity of the pipe
    PipePropertiesT // Customizable pipe properties
    >;

class flux_tempo_id;
using flux_tempo = pipe<
    flux_tempo_id,  // An identifier for the pipe
    unsigned int,   // The type of data in the pipe
    0,              // The capacity of the pipe
    PipePropertiesT // Customizable pipe properties
    >;

class second_trait;

template <typename flux_tempo, typename flux_out>
struct second_traitement
{

  sycl::ext::oneapi::experimental::annotated_arg<
      int, decltype(sycl::ext::oneapi::experimental::properties{
               stable})>
      taille_h;

  sycl::ext::oneapi::experimental::annotated_arg<
      int, decltype(sycl::ext::oneapi::experimental::properties{
               stable})>
      taille_v;

  auto get(sycl::ext::oneapi::experimental::properties_tag)
  {
    return sycl::ext::oneapi::experimental::properties{

        streaming_interface<>};
  }

  void operator()() const
  {
    [[intel::fpga_register]]
    unsigned int valeur_in,
        valeur_out;

    [[intel::initiation_interval(1)]]
    for (int px = 0; px < (taille_h) * (taille_v); px++)
    {
      valeur_in = flux_tempo::read();
      valeur_out = 255 - valeur_in;
      flux_out::write(valeur_out);
    }
  }
};

class voisionage;
template <typename flux_in, typename flux_tempo>
struct travail_sur_voisinage
{

  sycl::ext::oneapi::experimental::annotated_arg<
      int, decltype(sycl::ext::oneapi::experimental::properties{
               stable})>
      taille_h;

  sycl::ext::oneapi::experimental::annotated_arg<
      int, decltype(sycl::ext::oneapi::experimental::properties{
               stable})>
      taille_v;

  auto get(sycl::ext::oneapi::experimental::properties_tag)
  {
    return sycl::ext::oneapi::experimental::properties{

        streaming_interface<>};
  }

  void operator()() const
  {
    // Compteurs ligne pixel

    // Entree Sortie
    [[intel::fpga_register]]
    unsigned int pixel_a_traiter;

    [[intel::fpga_register]]
    unsigned int pixel_a_envoyer;

    //  [[intel::fpga_register]]
    //  int taille_h = TAILLE_IM-1;
    //  int taille_v = TAILLE_IM-1;

    [[intel::fpga_register]]
    unsigned int pixel_apres_traitement;

    // Ligne a retard
    [[intel::fpga_memory("BLOCK_RAM")]]
    unsigned int line_buffer[5][NB_COLONNE_MAX];

    // Voisinnage
    [[intel::fpga_register]]
    unsigned int fenetre[5][5];

    [[intel::initiation_interval(1)]]
    for (int num_lig = 0; (num_lig < NB_COLONNE_MAX) && num_lig < taille_v + 2; num_lig++)
    {

      [[intel::initiation_interval(1)]]
      //[[intel::speculated_iterations(0)]]
      for (int num_col = 0; (num_col < NB_COLONNE_MAX) && (num_col < taille_h + 2); num_col++)
      {

        if (num_lig < taille_v && num_col < taille_h)
        {

          pixel_a_traiter = flux_in::read();

          // Gestion ligne a retard

          fpga_tools::UnrolledLoop<0, 4>([&](auto l)
                                         { line_buffer[l][num_col] = line_buffer[l + 1][num_col]; });
          line_buffer[4][num_col] = pixel_a_traiter;

          // Fin gestion ligne a retard

          // Fenetre video glissante

          fpga_tools::UnrolledLoop<0, 5>([&](auto li)
                                         {
              // #pragma unroll

            fpga_tools::UnrolledLoop<0,4>([&](auto co)
            {
              fenetre[li][co] = fenetre[li][co + 1];
            });
            fenetre[li][4] = line_buffer[li][num_col]; });
          // Fin Fenetre video glissante
        }
        pixel_apres_traitement = traitement_5x5(fenetre);

        if ((num_lig >= 2) && (num_col >= 2))
        {
          pixel_a_envoyer = 0;

          if (((num_lig >= 4) && (num_lig < taille_v) && (num_col >= 4) && (num_col < taille_h)))
          {
            pixel_a_envoyer = pixel_apres_traitement;
          }
          flux_tempo::write(pixel_a_envoyer);
        }
      }
    }
  }
};

int main()
{
  try
  {

#if FPGA_SIMULATOR
    auto selector = sycl::ext::intel::fpga_simulator_selector_v;
    fstream fichier_out("sortie_sim.txt", ios::out);
#elif FPGA_HARDWARE
    auto selector = sycl::ext::intel::fpga_selector_v;
    fstream fichier_out("sortie_hw.txt", ios::out);
#else // #if FPGA_EMULATOR
    auto selector = sycl::ext::intel::fpga_emulator_selector_v;
    fstream fichier_out("sortie.txt", ios::out);
#endif
    sycl::queue q(selector, fpga_tools::exception_handler,
                  sycl::property::queue::enable_profiling{});

    fstream fichier_in("entree.txt", ios::in);

    unsigned int pixel = 0;
    int kernel_h = TAILLE_IM - 1;
    int kernel_v = TAILLE_IM - 1;

    cout << "Entrée" << endl;
    cout << "Launch kernel " << endl;
    q.single_task<voisionage>(travail_sur_voisinage<flux_in, flux_tempo>{kernel_h, kernel_v});
    q.single_task<second_trait>(second_traitement<flux_tempo, flux_out>{kernel_h, kernel_v});

    for (int l = 0; l < kernel_h; l++)
    {
      for (int c = 0; c < kernel_v; c++)
      {
        fichier_in >> pixel;
        flux_in::write(q, pixel);
      }
    }

    cout << "Sorties :" << endl;
    for (int l = 0; l < kernel_h; l++)
    {
      for (int c = 0; c < kernel_v; c++)
      {

        pixel = flux_out::read(q);
        fichier_out << pixel << " ";
      }
      fichier_out << endl;
    }

    q.wait();
    cout << "Fin de traitement" << endl;
  }
  catch (sycl::exception const &e)
  {
    std::cerr << "Caught a synchronous SYCL exception: " << e.what()
              << std::endl;

    std::terminate();
  }
}

unsigned int traitement_5x5(unsigned int entree[5][5])
{

  unsigned int sortie = 0;
  fpga_tools::UnrolledLoop<0, 5>([&](auto lig)
                                 { fpga_tools::UnrolledLoop<0, 5>([&](auto col)
                                                                  { sortie = sortie + entree[lig][col]; }); });
  sortie /= 25;

  return sortie;
}