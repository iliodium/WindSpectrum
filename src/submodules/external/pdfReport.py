from pydantic import validate_call
from src.common.annotation import ModelSizeType


class PdfReport:
    @staticmethod
    @validate_call
    def report(
            alpha: str,
            model_size: ModelSizeType
    ):
        """Создание отчёта для выбранной конфигурации.
        Отчёт включает:
            - отображение модели
            - изополя
            - огибающие
            - суммарные аэродинамические коэффициенты
            - суммарные аэродинамические коэффициенты в полярной системе координат
            - спектральная плотность мощности суммарных аэродинамических коэффициентов
            - характеристика по датчикам
        """

        folder = os.getcwd()

        model_scale, scale_factors = get_model_and_scale_factors(*model_size, alpha)
        breadth, depth, height = model_size
        name_report = f'Отчет ширина {breadth} глубина {depth} высота {height} альфа {alpha}'
        path_report = f'{folder}\\{name_report}'
        generate_directory_for_report(path_report)

        if model_scale[0] == model_scale[1]:
            angle_border = 50
        else:
            angle_border = 95

        x, z = self.clipboard_obj.get_coordinates(alpha, model_scale)
        args_welch_graphs = [(alpha, model_scale, str(angle))
                             for angle in range(0, 10, 5)]

        with ThreadPoolExecutor(max_workers=Core._count_threads) as executor:
            executor.map(lambda i: self.clipboard_obj.get_pressure_coefficients(*i), args_welch_graphs)

        # Отображение модели
        self.draw_model(alpha, model_size, model_scale, path_report)

        # Изополя
        self.draw_isofields(alpha, model_size, angle_border, path_report)

        # Огибающие
        self.draw_envelopes(alpha, model_scale, angle_border, path_report)

        # Суммарные аэродинамические коэффициенты в декартовой системе координат
        self.draw_summary_coefficients(alpha, model_size, angle_border, path_report)

        # Суммарные аэродинамические коэффициенты в полярной системе координат
        self.draw_summary_coefficients_polar(alpha, model_scale, model_size, angle_border, path_report)

        # Спектральная плотность мощности суммарных аэродинамических коэффициентов
        self.draw_welch_graphs(alpha, model_scale, model_size, angle_border, path_report)

        # Работа с word файлом
        doc = Document()
        style = doc.styles['Normal']
        style.font.size = Pt(14)
        style.font.name = 'Times New Roman'
        section = doc.sections[0]
        section.left_margin = Mm(30)
        section.right_margin = Mm(15)
        section.top_margin = Mm(20)
        section.bottom_margin = Mm(20)

        counter_plots = 1  # Счетчик графиков для нумерации
        counter_tables = 1  # Счетчик таблиц для нумерации

        title = doc.add_heading().add_run(f'Отчет по зданию {breadth}x{depth}x{height}')
        doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
        title.font.size = Pt(24)
        title.bold = True

        for i in ('Параметры ветрового воздействия:',
                  'Ветровой район: None',
                  'Тип местности: None'
                  ):
            doc.add_paragraph().add_run(i)
        p = doc.add_paragraph()

        run = p.add_run()
        run.add_picture(f'{path_report}\\Модель\\Модель 3D.png', width=Mm(82.5))
        run.add_picture(f'{path_report}\\Модель\\Модель в полярной системе координат.png', width=Mm(82.5))
        doc.add_paragraph().add_run(
            f'Рисунок {counter_plots}. Геометрические размеры и система координат направления ветровых потоков')
        counter_plots += 1

        doc.add_heading().add_run('1. Геометрические размеры здания').font.size = Pt(20)
        doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER

        header_model = (
            'Геометрический размер',
            'Значение, м'
        )
        table_model = doc.add_table(rows=1, cols=len(header_model))
        table_model.style = 'Table Grid'
        hdr_cells = table_model.rows[0].cells
        for i in range(len(header_model)):
            hdr_cells[i].add_paragraph().add_run(header_model[i])
        for i, j in zip((breadth, depth, height), ('Ширина:', 'Глубина:', 'Высота:')):
            row_cells = table_model.add_row().cells
            row_cells[0].add_paragraph().add_run(j)
            row_cells[1].add_paragraph().add_run(str(i))
        doc.add_picture(f'{path_report}\\Модель\\Развертка модели.png', width=Mm(165))
        doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
        doc.add_paragraph().add_run(f'Рисунок {counter_plots}. Система датчиков мониторинга')
        counter_plots += 1
        doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
        doc.add_page_break()


        doc.add_heading().add_run('2. Статистика по датчиках. Максимумы и огибающие').font.size = Pt(20)
        doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER

        for angle in range(0, angle_border, 5):
            envelopes = glob.glob(
                f'{path_report}\\Огибающие\\Огибающие {model_scale} {alpha} {angle:02}\\Огибающие *.png')
            for i in envelopes:
                doc.add_picture(i, height=Mm(80))
                doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
                doc.add_paragraph().add_run(
                    f'Рисунок {counter_plots}. Огибающие ветрового давления для здания '
                    f'{breadth}x{depth}x{height} угол {angle:02}º '
                    f'датчики {i[i[:i.rfind("-") - 1].rfind(" ") + 1:i.rfind(".")]}')
                doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
                counter_plots += 1
        doc.add_page_break()


        doc.add_heading().add_run('3. Изополя ветровых нагрузок и воздействий').font.size = Pt(20)
        doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER


        doc.add_heading(level=2).add_run('3.1 Непрерывные изополя').font.size = Pt(16)
        doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER

        mods = ('MAX', 'MEAN', 'MIN', 'STD')
        for mode in mods:
            for angle in range(0, angle_border, 5):
                isofields = f'{path_report}\\Изополя ветровых нагрузок и воздействий\\Непрерывные\\{mode}\\' \
                            f'Изополя {breadth} {depth} {height} {alpha} {angle:02} {mode}.png'

                doc.add_picture(isofields)
                doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
                doc.add_paragraph().add_run(
                    f'Рисунок {counter_plots}. Непрерывные изополя {mode} '
                    f'для здания {breadth}x{depth}x{height} угол {angle:02}º')
                doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
                counter_plots += 1

        doc.add_heading(level=2).add_run('3.2 Дискретные изополя').font.size = Pt(16)
        doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER

        for mode in mods:
            for angle in range(0, angle_border, 5):
                isofields = f'{path_report}\\Изополя ветровых нагрузок и воздействий\\Дискретные\\{mode}\\' \
                            f'Изополя {breadth} {depth} {height} {alpha} {angle:02} {mode}.png'

                doc.add_picture(isofields)
                doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
                doc.add_paragraph().add_run(
                    f'Рисунок {counter_plots}. Непрерывные изополя {mode} '
                    f'для здания {breadth}x{depth}x{height} угол {angle:02}º')
                doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
                counter_plots += 1
        doc.add_page_break()

        # coordinates = converter_coordinates_to_real(x, z, model_size, model_scale)
        # sensor_statistics = self.get_sensor_statistics(alpha,
        #                                                model_scale,
        #                                                angle_border,
        #                                                model_size,
        #                                                coordinates,
        #                                                )
        #
        # doc.add_heading().add_run('4. Статистика по датчикам в табличном виде ').font.size = Pt(20)
        # doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
        # header_sensors = (
        #     'ДАТЧИК',
        #     'X(мм)',
        #     'Y(мм)',
        #     'Z(мм)',
        #     'MEAN',
        #     'RMS',
        #     'STD',
        #     'MAX',
        #     'MIN',
        #     'РАСЧЕТНОЕ',
        #     'ОБЕСП+',
        #     'ОБЕСП-'
        # )
        # for angle in range(0, angle_border, 5):
        #     doc.add_paragraph().add_run(
        #         f'\nТаблица {counter_tables}. Аэродинамический коэффициент в датчиках для '
        #         f'здания {breadth}x{depth}x{height} угол {angle:02}º')
        #     doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
        #     counter_tables += 1
        #     table_sensors = doc.add_table(rows=1, cols=len(header_sensors))
        #     table_sensors.style = 'Table Grid'
        #     hdr_cells = table_sensors.rows[0].cells
        #     for i in range(len(header_sensors)):
        #         hdr_cells[i].add_paragraph().add_run(header_sensors[i]).font.size = Pt(8)
        #
        #     for rec in sensor_statistics[angle // 5]:
        #         row_cells = table_sensors.add_row().cells
        #         for i in range(len(rec)):
        #             row_cells[i].add_paragraph().add_run(str(rec[i])).font.size = Pt(12)
        # doc.add_page_break()
        #
        # del sensor_statistics


        # summary_coefficients_statistics = self.get_summary_coefficients_statistics(angle_border,
        # alpha,
        # model_scale,
        # )

        doc.add_heading().add_run('5. Суммарные значения аэродинамических коэффициентов').font.size = Pt(20)
        doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
        header_sum = (
            'СИЛА',
            'MEAN',
            'RMS',
            'STD',
            'MAX',
            'MIN',
            'РАСЧЕТНОЕ',
            'ОБЕСП+',
            'ОБЕСП-'
        )
        for angle in range(0, angle_border, 5):

            doc.add_picture(
                f'{path_report}\\Суммарные аэродинамические коэффициенты\\Декартовая система координат\\'
                f'Суммарные аэродинамические коэффициенты Cx_Cy_CMz {" ".join(model_size)} {alpha} {angle:02}.png')
            doc.add_paragraph().add_run(
                f'Рисунок {counter_plots}. Суммарные аэродинамические коэффициенты '
                f'для здания {breadth}x{depth}x{height} угол {angle:02}º')
            doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
            counter_plots += 1
            # doc.add_paragraph().add_run(
            #     f'Таблица {counter_tables}. Суммарные аэродинамические коэффициенты '
            #     f'для здания {breadth}x{depth}x{height} угол {angle:02}º')
            # counter_tables += 1
            # table_sum = doc.add_table(rows=1, cols=len(header_sum))
            # table_sum.style = 'Table Grid'
            # hdr_cells = table_sum.rows[0].cells
            # for i in range(len(header_sum)):
            #     hdr_cells[i].add_paragraph().add_run(header_sum[i]).font.size = Pt(8)
            #
            # for rec in summary_coefficients_statistics[angle // 5]:
            #     row_cells = table_sum.add_row().cells
            #     for i in range(len(rec)):
            #         row_cells[i].add_paragraph().add_run(str(rec[i])).font.size = Pt(12)

            doc.add_page_break()

        for mode in header_sum[1:]:
            doc.add_picture(f'{path_report}\\Суммарные аэродинамические коэффициенты\\Полярная система координат\\'
                            f'Суммарные аэродинамические коэффициенты Cx Cy CMz {breadth} {depth} {height} {mode} '
                            f'в полярной системе координат.png')

            doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
            doc.add_paragraph().add_run(
                f'Рисунок {counter_plots}. Суммарные аэродинамические коэффициенты в полярной системе координат {mode}'
                f' для здания {breadth}x{depth}x{height}')
            doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
            counter_plots += 1
            doc.add_page_break()

        # del summary_coefficients_statistics
        doc.add_heading().add_run('6. Спектры cуммарных значений аэродинамических коэффициентов').font.size = Pt(20)
        doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
        for angle in range(0, angle_border, 5):

            doc.add_picture(f'{path_report}\\Спектральная плотность мощности\\Логарифмическая шкала\\'
                            f'Спектральная плотность мощности {model_scale} {alpha} {angle:02}.png')
            doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
            doc.add_paragraph().add_run(
                f'Рисунок {counter_plots}. Спектр cуммарных значений аэродинамических коэффициентов '
                f'для здания {breadth}x{depth}x{height} угол {angle:02} º')
            doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
            counter_plots += 1
        doc.add_page_break()

        doc.save(f'{path_report}\\Отчет ширина {breadth} глубина {depth} высота {height} альфа {alpha}.docx')
        # os.startfile(f'{path_report}\\Отчет ширина {breadth} глубина {depth} высота {height} альфа {alpha}.docx')
